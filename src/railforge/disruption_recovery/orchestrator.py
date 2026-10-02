"""Recovery orchestration: decide safely, publish atomically, roll back cleanly.

The orchestrator is the only place that mutates operational state.  Its rules:

1. enumerate structural alternatives (including unsafe ones, so their refusal
   is on the record);
2. apply the hard safety gate — **no deadline can override a hard finding**;
3. apply the published risk appetite — an on-time but uncontrolled alternative
   is refused just like an unsafe one;
4. rank what remains purely by customer harm and residual risk;
5. publish plan + power/route/crew reservations in one transaction.  Any
   failure (version conflict, contention, publication hook) rolls the whole
   transaction back; state is unchanged and every reason is preserved in the
   audit ledger;
6. if nothing controlled exists, no plan is published; the train is held and
   the decision escalated with the full refusal reasons attached.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from . import candidates as candidates_mod
from . import evaluator as evaluator_mod
from .atomic import (
    PublicationFailed,
    ResourceContention,
    ResourceStore,
    VersionConflict,
    stage_reservation,
)
from .domain import (
    AuditCode,
    DecisionCode,
    Evaluation,
    RecoveryPlan,
    RecoveryRequest,
)


def utc(at: datetime | None) -> datetime:
    if at is None:
        return datetime.now(timezone.utc)
    if at.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return at.astimezone(timezone.utc)


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(v) for v in value)
    return value


@dataclass(frozen=True)
class AuditEntry:
    code: AuditCode
    subject: str
    at: datetime
    detail: Mapping[str, Any]


class InMemoryAuditLedger:
    """Append-only ledger.  Entries survive rollbacks — that is the point."""

    def __init__(self):
        self._entries: list[AuditEntry] = []

    def append(self, code: AuditCode, subject: str, at: datetime,
               detail: Mapping[str, Any] | None = None) -> AuditEntry:
        entry = AuditEntry(code, subject, utc(at), dict(detail or {}))
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def codes(self) -> list[AuditCode]:
        return [e.code for e in self._entries]


@dataclass(frozen=True)
class RecoveryOutcome:
    request_id: str
    decision: DecisionCode
    plan: RecoveryPlan | None
    evaluations: tuple[Evaluation, ...]
    rollback_reason: str | None
    audit: tuple[AuditEntry, ...]


PublicationHook = Callable[[dict[str, Any], dict[str, Any]], None]


class RecoveryOrchestrator:
    def __init__(self, store: ResourceStore | None = None,
                 ledger: InMemoryAuditLedger | None = None,
                 publication_hook: PublicationHook | None = None):
        self.store = store or ResourceStore()
        self.ledger = ledger or InMemoryAuditLedger()
        self.publication_hook = publication_hook
        self._outcomes: dict[str, RecoveryOutcome] = {}

    # ------------------------------------------------------------------ run
    def run(self, request: RecoveryRequest) -> RecoveryOutcome:
        # Idempotent replay: a given request id resolves to exactly one outcome.
        previous = self._outcomes.get(request.request_id)
        if previous is not None:
            return previous
        outcome = self._decide_and_publish(request)
        self._outcomes[request.request_id] = outcome
        return outcome

    def _decide_and_publish(self, request: RecoveryRequest) -> RecoveryOutcome:
        at = utc(request.initiated_at)
        train = request.train
        audits: list[AuditEntry] = []

        def audit(code: AuditCode, subject: str, **detail: Any) -> None:
            audits.append(self.ledger.append(
                code, subject, at, jsonable(detail)))

        audit(AuditCode.REQUEST_RECEIVED, request.request_id,
              request_id=request.request_id, train_id=train.train_id,
              origin=train.origin, destination=train.destination,
              cars=len(train.cars), gross_t=train.gross_tonnes(),
              hazmat=train.has_hazmat(),
              blockage_segment=train.blockage.segment_id,
              max_acceptable_risk=request.max_acceptable_risk)

        options = candidates_mod.generate(request)
        evaluations: list[Evaluation] = []
        for option in options:
            ev = evaluator_mod.evaluate_option(request, option)
            evaluations.append(ev)
            if ev.hard_rejections:
                audit(AuditCode.OPTION_REJECTED_SAFETY, option.option_id,
                      kind=option.kind.value, path=list(option.path),
                      findings=[asdict(f) for f in ev.hard_rejections])
            elif ev.risk_findings:
                audit(AuditCode.OPTION_REJECTED_RISK, option.option_id,
                      kind=option.kind.value, path=list(option.path),
                      risk_score=ev.risk_score,
                      findings=[asdict(f) for f in ev.risk_findings])
            elif ev.commitment_findings:
                audit(AuditCode.OPTION_REJECTED_COMMITMENT, option.option_id,
                      kind=option.kind.value, path=list(option.path),
                      risk_score=ev.risk_score,
                      max_lateness_minutes=ev.max_lateness_minutes,
                      broken_commitments=list(ev.broken_commitments),
                      findings=[asdict(f) for f in ev.commitment_findings])
            else:
                audit(AuditCode.OPTION_SCORED, option.option_id,
                      kind=option.kind.value, path=list(option.path),
                      risk_score=ev.risk_score,
                      max_lateness_minutes=ev.max_lateness_minutes,
                      broken_commitments=list(ev.broken_commitments))

        selected = evaluator_mod.select(evaluations)
        if selected is None:
            # Controlled hold + escalation.  Nothing is published and the
            # fleet is not moved into an unsafe or uncontrolled state.
            reasons = self._refusal_reasons(evaluations)
            audit(AuditCode.RECOVERY_ABORTED, request.request_id,
                  decision=DecisionCode.NO_CONTROLLED_ALTERNATIVE.value,
                  reasons=reasons,
                  alternatives_considered=len(evaluations))
            return RecoveryOutcome(
                request_id=request.request_id,
                decision=DecisionCode.NO_CONTROLLED_ALTERNATIVE,
                plan=None,
                evaluations=tuple(evaluations),
                rollback_reason=None,
                audit=tuple(audits),
            )

        option = selected.option
        plan_id = f"plan-{request.request_id}"
        audit(AuditCode.OPTION_SELECTED, option.option_id,
              plan_id=plan_id, kind=option.kind.value, path=list(option.path),
              risk_score=selected.risk_score,
              max_lateness_minutes=selected.max_lateness_minutes)

        plan = RecoveryPlan(
            plan_id=plan_id,
            request_id=request.request_id,
            train_id=train.train_id,
            option_id=option.option_id,
            kind=option.kind,
            path=option.path,
            power_ids=selected.itinerary.power_ids,
            crew_changes=selected.itinerary.crew_changes,
            depart_at=selected.itinerary.depart_at,
            arrive_at=selected.itinerary.arrive_at,
            movements=selected.itinerary.movements,
            risk_score=selected.risk_score,
            max_lateness_minutes=selected.max_lateness_minutes,
            published_at=at,
        )

        return self._publish(request, plan, evaluations, audits, at)

    # ------------------------------------------------------------- publishing
    def _publish(self, request: RecoveryRequest, plan: RecoveryPlan,
                 evaluations: list[Evaluation], audits: list[AuditEntry],
                 at: datetime) -> RecoveryOutcome:
        shadow = self.store.begin()
        record = jsonable(plan)

        try:
            # Stage every reservation the plan needs — all or nothing.
            stage_reservation(
                shadow, self.store, "train", plan.train_id, plan.plan_id,
                {"state": "recovery_published", "path": list(plan.path)})
            for loco_id in plan.power_ids:
                stage_reservation(
                    shadow, self.store, "locomotive", loco_id, plan.plan_id,
                    {"assigned_to": plan.train_id})
            for segment_id in {m.segment.segment_id for m in plan.movements}:
                stage_reservation(
                    shadow, self.store, "route_segment", segment_id,
                    plan.plan_id, {"train": plan.train_id})

            publication = self.store.commit(
                shadow, plan.plan_id, record, at,
                publication_hook=self._wrap_hook())
        except (VersionConflict, ResourceContention, PublicationFailed) as exc:
            reason = self._describe_failure(exc)
            report = self.store.rollback(
                shadow, plan.plan_id, reason, at)
            audits.append(self.ledger.append(
                AuditCode.PLAN_PUBLISH_FAILED, plan.plan_id, at,
                {"reason": reason, "touched": list(report.touched)}))
            audits.append(self.ledger.append(
                AuditCode.PLAN_ROLLED_BACK, plan.plan_id, at,
                {"state_unchanged": report.state_unchanged,
                 "decision": DecisionCode.ROLLED_BACK.value,
                 "touched": list(report.touched)}))
            decision = (DecisionCode.PUBLICATION_CONFLICT
                        if isinstance(exc, VersionConflict)
                        else DecisionCode.PUBLICATION_FAILED)
            return RecoveryOutcome(
                request_id=request.request_id,
                decision=decision,
                plan=None,
                evaluations=tuple(evaluations),
                rollback_reason=reason,
                audit=tuple(audits),
            )

        audits.append(self.ledger.append(
            AuditCode.PLAN_COMMITTED, plan.plan_id, at,
            {"reservations": list(publication["reservations"]),
             "risk_score": plan.risk_score,
             "max_lateness_minutes": plan.max_lateness_minutes,
             "decision": DecisionCode.PUBLISHED.value}))
        return RecoveryOutcome(
            request_id=request.request_id,
            decision=DecisionCode.PUBLISHED,
            plan=plan,
            evaluations=tuple(evaluations),
            rollback_reason=None,
            audit=tuple(audits),
        )

    def _wrap_hook(self) -> PublicationHook | None:
        hook = self.publication_hook
        if hook is None:
            return None

        def wrapped(publication: dict[str, Any], resources: dict[str, Any]) -> None:
            hook(publication, resources)

        return wrapped

    # ------------------------------------------------------------------ misc
    @staticmethod
    def _describe_failure(exc: Exception) -> str:
        if isinstance(exc, PublicationFailed):
            # atomic.py attaches the RollbackReport as the second arg.
            report = exc.args[1] if len(exc.args) > 1 else None
            if report is not None:
                return report.reason
            return str(exc)
        if isinstance(exc, VersionConflict):
            return f"optimistic version conflict: {exc}"
        if isinstance(exc, ResourceContention):
            return f"resource contention: {exc}"
        return f"publication failed: {exc}"

    @staticmethod
    def _refusal_reasons(evaluations: list[Evaluation]) -> list[dict[str, Any]]:
        reasons: list[dict[str, Any]] = []
        for ev in evaluations:
            if ev.hard_rejections:
                rejected_by, blocking = "safety", ev.hard_rejections
            elif ev.risk_findings:
                rejected_by, blocking = "risk", ev.risk_findings
            elif ev.commitment_findings:
                rejected_by, blocking = "commitment", ev.commitment_findings
            else:
                continue
            reasons.append({
                "option_id": ev.option.option_id,
                "kind": ev.option.kind.value,
                "path": list(ev.option.path),
                "rejected_by": rejected_by,
                "codes": sorted({f.code for f in blocking}),
                "messages": [f.message for f in blocking],
            })
        return reasons
