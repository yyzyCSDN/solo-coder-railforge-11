"""Saga orchestration for freight disruption recovery.

Staging flow:

    select plan  ->  stage resources  ->  atomic control publish  ->  commit

Every step before publication is provisional.  When the plan cannot be
selected, resources are already held elsewhere, or publication itself fails,
every staged lease is released, a hash-chained audit record is appended and
the structured reasons are retained in an append-only rollback journal.
Committed safety constraints live in ``SafetyPolicy`` and are never relaxed
to rescue local punctuality.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from threading import RLock
from typing import Mapping

from railforge.audit.trail import Trail
from railforge.core.time import utc
from railforge.events.stream import EventStream
from railforge.recovery_saga.model import (
    AffectedMove,
    DisruptionInput,
    MALFORMED_INPUT,
    RecoveryInfeasible,
    RecoveryLocomotive,
    RecoveryPlan,
    Rejection,
    SafetyPolicy,
    build_plan,
)
from railforge.storage.ops_store import VersionedStore

RECOVERY_TOPIC = "recovery.plan.published"
ROLLBACK_ACTION = "recovery.plan.rolled_back"
ROLLBACK_ACTOR = "railforge-recovery-saga"

NO_SAFE_PLAN = "no_safe_plan"
RESOURCE_CONTENTION = "resource_contention"
PUBLICATION_FAILED = "publication_failed"
INVALID_REQUEST = "invalid_request"

WAGON = "wagon"
LOCOMOTIVE = "locomotive"


@dataclass(frozen=True)
class Lease:
    operation_id: str
    resource_kind: str
    resource_id: str
    move_id: str


class ResourceContention(Exception):
    """A staged resource is already held by another recovery operation."""

    def __init__(self, lease: Lease, holder: str):
        self.lease = lease
        self.holder = holder
        super().__init__(
            f"{lease.resource_kind} {lease.resource_id} held by {holder}")


class ResourceLedger:
    """Two-phase lease ledger: resources are staged then committed or released."""

    def __init__(self):
        self._staged: dict[str, list[Lease]] = {}
        self._holders: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    def stage(self, operation_id: str, resource_kind: str,
              resource_id: str, move_id: str) -> Lease:
        lease = Lease(operation_id, resource_kind, resource_id, move_id)
        with self._lock:
            key = (resource_kind, resource_id)
            holder = self._holders.get(key)
            if holder is not None and holder != operation_id:
                raise ResourceContention(lease, holder)
            if holder == operation_id:
                return lease
            self._holders[key] = operation_id
            self._staged.setdefault(operation_id, []).append(lease)
            return lease

    def rollback(self, operation_id: str) -> tuple[Lease, ...]:
        """Release everything still staged by this operation."""
        with self._lock:
            leases = tuple(self._staged.pop(operation_id, ()))
            for lease in leases:
                if self._holders.get((lease.resource_kind,
                                      lease.resource_id)) == operation_id:
                    del self._holders[(lease.resource_kind, lease.resource_id)]
            return leases

    def commit(self, operation_id: str) -> tuple[Lease, ...]:
        with self._lock:
            return tuple(self._staged.pop(operation_id, ()))

    def held_by(self, resource_kind: str, resource_id: str) -> str | None:
        with self._lock:
            return self._holders.get((resource_kind, resource_id))


@dataclass(frozen=True)
class RollbackRecord:
    operation_id: str
    disruption_id: str
    reason_code: str
    rejections: tuple[Rejection, ...]
    released: tuple[Lease, ...]
    audit_index: int
    audit_digest: str
    at: datetime


@dataclass(frozen=True)
class SagaJournalEntry:
    operation_id: str
    disruption_id: str
    status: str  # "committed" or "rolled_back"
    at: datetime
    audit_index: int
    audit_digest: str
    plan: RecoveryPlan | None = None
    rollback: RollbackRecord | None = None


class RecoveryAborted(Exception):
    """The recovery saga failed; all side effects were rolled back."""

    def __init__(self, record: RollbackRecord):
        self.record = record
        self.operation_id = record.operation_id
        self.reason_code = record.reason_code
        self.rejections = record.rejections
        super().__init__(
            f"recovery {record.operation_id} rolled back: {record.reason_code}")


class RecoveryOrchestrator:
    """Coordinates selection, atomic publication and compensating rollback."""

    def __init__(self, control=None, ledger: ResourceLedger | None = None,
                 stream: EventStream | None = None, trail: Trail | None = None,
                 store: VersionedStore | None = None):
        if control is None:
            from railforge.workflows.control import RailForgeWorkflowControl

            control = RailForgeWorkflowControl(stream=stream, trail=trail,
                                               store=store)
        self.control = control
        self.trail = control.trail
        self.ledger = ledger or ResourceLedger()
        self._lock = RLock()
        self._journal: list[SagaJournalEntry] = []
        self._rollbacks: dict[str, RollbackRecord] = {}
        self._committed: dict[str, object] = {}

    @property
    def journal(self) -> tuple[SagaJournalEntry, ...]:
        with self._lock:
            return tuple(self._journal)

    @property
    def rollbacks(self) -> Mapping[str, RollbackRecord]:
        with self._lock:
            return dict(self._rollbacks)

    def _audit(self, disruption_id: str, at: datetime):
        return self.trail.append(ROLLBACK_ACTOR, ROLLBACK_ACTION,
                                 disruption_id, at)

    def _abort(self, operation_id, disruption_id, reason_code, rejections,
               at) -> RecoveryAborted:
        """Must be called while nothing is committed: release and record."""
        released = self.ledger.rollback(operation_id)
        record_audit = self._audit(disruption_id, at)
        record = RollbackRecord(
            operation_id=operation_id,
            disruption_id=disruption_id,
            reason_code=reason_code,
            rejections=tuple(rejections),
            released=released,
            audit_index=record_audit.index,
            audit_digest=record_audit.digest,
            at=at,
        )
        self._rollbacks[operation_id] = record
        self._journal.append(SagaJournalEntry(
            operation_id=operation_id,
            disruption_id=disruption_id,
            status="rolled_back",
            at=at,
            audit_index=record_audit.index,
            audit_digest=record_audit.digest,
            rollback=record,
        ))
        return RecoveryAborted(record)

    def publish(self,
                operation_id: str,
                disruption: DisruptionInput,
                moves: list[AffectedMove],
                vehicles,
                locomotives: list[RecoveryLocomotive],
                at: datetime,
                policy: SafetyPolicy | None = None):
        """Select and atomically publish the recovery plan.

        Returns the control-plane ``OperationReceipt`` on success or raises
        :class:`RecoveryAborted` after a complete rollback.  Replaying an
        operation id returns the original outcome and never double-publishes.
        """
        at = utc(at)
        with self._lock:
            if operation_id in self._committed:
                return self._committed[operation_id]
            prior = self._rollbacks.get(operation_id)
            if prior is not None:
                raise RecoveryAborted(prior)

            try:
                plan = build_plan(disruption, moves, vehicles, locomotives,
                                  policy)
            except RecoveryInfeasible as exc:
                raise self._abort(operation_id, disruption.disruption_id,
                                  NO_SAFE_PLAN, exc.rejections, at)
            except ValueError as exc:
                rejection = Rejection("-", None, MALFORMED_INPUT, str(exc))
                raise self._abort(operation_id, disruption.disruption_id,
                                  INVALID_REQUEST, (rejection,), at)

            try:
                for move in moves:
                    for wagon in sorted(move.wagons):
                        self.ledger.stage(operation_id, WAGON, wagon,
                                         move.move_id)
                for loco_id in sorted(plan.locomotives):
                    move_id = next(
                        a.move_id for a in plan.assignments
                        if loco_id in a.locomotives)
                    self.ledger.stage(operation_id, LOCOMOTIVE, loco_id,
                                      move_id)
            except ResourceContention as exc:
                rejection = Rejection(
                    exc.lease.move_id, None, exc.lease.resource_kind +
                    "_resource_contention",
                    f"{exc.lease.resource_kind} {exc.lease.resource_id} "
                    f"already assigned to recovery {exc.holder}",
                    resource=exc.lease.resource_id)
                raise self._abort(operation_id, disruption.disruption_id,
                                  RESOURCE_CONTENTION, (rejection,), at)

            published = replace(plan, published_at=at)
            evidence = {
                "disruption_id": disruption.disruption_id,
                "affected_paths": list(disruption.affected_paths),
                "moves": len(moves),
                "safety_hold": True,
            }
            try:
                receipt = self.control.run(
                    operation_id, RECOVERY_TOPIC, disruption.disruption_id, at,
                    lambda: published, evidence)
            except Exception as exc:  # publication is all-or-nothing
                rejection = Rejection("-", None, PUBLICATION_FAILED, str(exc))
                raise self._abort(operation_id, disruption.disruption_id,
                                  PUBLICATION_FAILED, (rejection,), at)

            self.ledger.commit(operation_id)
            self._committed[operation_id] = receipt
            self._journal.append(SagaJournalEntry(
                operation_id=operation_id,
                disruption_id=disruption.disruption_id,
                status="committed",
                at=at,
                audit_index=receipt.audit_index,
                audit_digest=self._digest_for(receipt.audit_index),
                plan=published,
            ))
            return receipt

    def _digest_for(self, audit_index: int) -> str:
        record = self.trail.record(audit_index)
        return record.digest if record is not None else ""
