"""Atomic publication primitives for recovery plans.

A plan is published through a small transaction that follows copy-on-write
optimistic concurrency:

1. read the versions of every resource the plan touches;
2. stage reservations and the plan record on a private shadow copy;
3. inside the commit lock, re-check versions — a concurrent change raises
   ``VersionConflict``;
4. run the publication hook (outbox/event sink); if it raises, the shadow is
   discarded and the committed state is byte-for-byte unchanged;
5. only then swap the shadow into the store and bump versions.

There is no partial state: either the plan *and* every reservation become
visible together, or nothing does.  A failed transaction returns a rolled-back
snapshot description so callers can retain the reason in the audit ledger.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any, Callable


class VersionConflict(RuntimeError):
    """Raised when a resource changed between read and commit."""


class PublicationFailed(RuntimeError):
    """Raised by a publication hook; the transaction is rolled back."""


class ResourceContention(RuntimeError):
    """Staged resource is already reserved by another plan."""


@dataclass
class Resource:
    kind: str
    resource_id: str
    version: int = 0
    # Reservation owner plan id, or None when free.
    reserved_by: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackReport:
    plan_id: str
    reason: str
    touched: tuple[str, ...]
    state_unchanged: bool
    at: datetime


class _Shadow:
    def __init__(self, base: "ResourceStore"):
        self.resources: dict[str, Resource] = {}
        for key, res in base.resources.items():
            clone = copy.copy(res)
            clone.data = copy.deepcopy(res.data)
            self.resources[key] = clone
        # Version of every live resource at begin; a key absent here (and in
        # the live ledger) is treated as expected version 0.
        self.expected_versions = {k: r.version for k, r in base.resources.items()}
        self.plan_records: dict[str, Any] = copy.deepcopy(base.plan_records)
        self.touched: list[str] = []
        self.created: list[str] = []

    @staticmethod
    def key(kind: str, resource_id: str) -> str:
        return f"{kind}:{resource_id}"


class ResourceStore:
    """Versioned resource ledger used as the publication target."""

    def __init__(self):
        self.resources: dict[str, Resource] = {}
        self.plan_records: dict[str, Any] = {}
        self._lock = RLock()

    # -- reads -------------------------------------------------------------
    def ensure(self, kind: str, resource_id: str, data: dict[str, Any] | None = None) -> Resource:
        with self._lock:
            key = f"{kind}:{resource_id}"
            res = self.resources.get(key)
            if res is None:
                res = Resource(kind=kind, resource_id=resource_id, data=dict(data or {}))
                self.resources[key] = res
            return res

    def get(self, kind: str, resource_id: str) -> Resource | None:
        with self._lock:
            return self.resources.get(f"{kind}:{resource_id}")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "resources": {
                    k: {"version": r.version, "reserved_by": r.reserved_by,
                        "data": copy.deepcopy(r.data)}
                    for k, r in sorted(self.resources.items())
                },
                "plans": copy.deepcopy(self.plan_records),
            }

    # -- transaction -------------------------------------------------------
    def begin(self) -> _Shadow:
        with self._lock:
            shadow = _Shadow(self)
            return shadow

    def commit(self, shadow: _Shadow, plan_id: str, plan_record: dict[str, Any],
               at: datetime,
               publication_hook: Callable[[dict[str, Any], dict[str, Resource]], None] | None = None
               ) -> dict[str, Any]:
        with self._lock:
            # Optimistic re-read: every resource that existed at begin() must
            # still be at the same version.  Resources created inside the
            # transaction must still be absent.
            for key, expected in shadow.expected_versions.items():
                if key in shadow.created:
                    continue
                current = self.resources.get(key)
                if current is None or current.version != expected:
                    raise VersionConflict(
                        f"{key} changed: expected version {expected}")
            for key in shadow.created:
                if key in self.resources:
                    raise VersionConflict(
                        f"{key} created concurrently while staging")

            # The staged plan and resources form the single publication unit.
            publication = {
                "plan_id": plan_id,
                "plan": plan_record,
                "reservations": {
                    k: {"kind": r.kind, "resource_id": r.resource_id}
                    for k, r in shadow.resources.items()
                    if r.reserved_by == plan_id
                },
                "at": at,
            }

            try:
                if publication_hook is not None:
                    # Hook gets a frozen view; it cannot mutate ledger state.
                    publication_hook(publication, dict(shadow.resources))
            except Exception as exc:  # publication side-effect failed -> rollback
                report = RollbackReport(
                    plan_id=plan_id,
                    reason=f"publication hook failed: {exc}",
                    touched=tuple(shadow.touched),
                    state_unchanged=True,
                    at=at,
                )
                raise PublicationFailed(str(exc), report) from exc

            # Commit point: swap staged resources in as a block.  Nothing in
            # the live ledger changed before this loop.
            for key in shadow.touched:
                staged = shadow.resources[key]
                if key in shadow.created:
                    if key in self.resources:
                        raise VersionConflict(f"{key} created concurrently at commit")
                    staged.version = 1
                else:
                    live = self.resources.get(key)
                    expected = shadow.expected_versions.get(key)
                    if live is None or expected is None or live.version != expected:
                        raise VersionConflict(f"{key} changed during commit")
                    staged.version = expected + 1
                self.resources[key] = staged

            self.plan_records[plan_id] = plan_record
            return publication

    def rollback(self, shadow: _Shadow, plan_id: str, reason: str,
                 at: datetime) -> RollbackReport:
        # Dropping the shadow leaves the live ledger untouched.  The report is
        # the only artifact callers keep — together with the audit trail.
        return RollbackReport(
            plan_id=plan_id,
            reason=reason,
            touched=tuple(shadow.touched),
            state_unchanged=True,
            at=at,
        )


# -- staging helpers used by the orchestrator --------------------------------

def stage_reservation(shadow: _Shadow, store: ResourceStore, kind: str,
                      resource_id: str, plan_id: str,
                      data: dict[str, Any] | None = None) -> None:
    """Reserve a resource on the shadow only.

    The live ledger is never touched here: a resource that does not yet exist
    is materialized inside the transaction and inserted atomically at commit.
    """
    key = shadow.key(kind, resource_id)
    res = shadow.resources.get(key)
    if res is None:
        live = store.get(kind, resource_id)  # read-only lookup, no mutation
        if live is None:
            res = Resource(kind=kind, resource_id=resource_id, data=dict(data or {}))
            shadow.created.append(key)
        else:
            res = copy.copy(live)
            res.data = copy.deepcopy(live.data)
            shadow.expected_versions[key] = live.version
        shadow.resources[key] = res
    if res.reserved_by is not None and res.reserved_by != plan_id:
        raise ResourceContention(
            f"{key} already reserved by {res.reserved_by}")
    res.reserved_by = plan_id
    if data:
        res.data.update(data)
    if key not in shadow.touched:
        shadow.touched.append(key)
