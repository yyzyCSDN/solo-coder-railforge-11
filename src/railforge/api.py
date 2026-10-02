from __future__ import annotations

from railforge.braking.brake_percentage import brake_percentage
from railforge.consist.integrity import validate
from railforge.disruption_recovery.orchestrator import (
    RecoveryOrchestrator,
    jsonable,
    utc,
)
from railforge.routing.clearance import route_ok
from railforge.signaling.block_occupancy import conflicts
from railforge.workflows.control import RailForgeWorkflowControl
from railforge.workflows.features import RailForgeFeatureService
from railforge.capability_catalog import load_capabilities


class RailForgeService:
    def __init__(self, control: RailForgeWorkflowControl | None = None,
                 recovery: RecoveryOrchestrator | None = None):
        self.control = control or RailForgeWorkflowControl()
        self.features = RailForgeFeatureService(self.control)
        self._recovery = recovery or RecoveryOrchestrator()

    def health(self):
        return {
            "service": "railforge",
            "status": "ok",
            "runtime_dependencies": 0,
            "capabilities": load_capabilities(),
            "workflow": {"events": self.control.stream.head, "audit_valid": self.control.trail.verify()},
        }

    def block_conflicts(self, candidate, existing, clearance):
        return conflicts(candidate, existing, clearance)

    def validate_consist(self, rows):
        return validate(rows)

    def brake_percentage(self, rows, gradient):
        return brake_percentage(rows, gradient)

    def route_clearance(self, profile, segments):
        return route_ok(profile, segments)

    def workflow(self):
        return self.control

    def recovery_orchestrator(self) -> RecoveryOrchestrator:
        return self._recovery

    def recover_disruption(self, request):
        """Orchestrate a recovery and project the terminal decision onto the
        common control plane.

        The projection is a recording only: on rollback / refusal the
        orchestrator's resource ledger is untouched and the event carries the
        retained reason; the recovery audit ledger holds every refusal detail.
        """
        outcome = self._recovery.run(request)
        at = utc(request.initiated_at)
        plan = jsonable(outcome.plan) if outcome.plan is not None else None
        summary = {
            "decision": outcome.decision.value,
            "plan": plan,
            "rollback_reason": outcome.rollback_reason,
            "alternatives_considered": len(outcome.evaluations),
            "audit_entries": len(outcome.audit),
        }
        receipt = self.control.run(
            f"disruption-recovery:{request.request_id}",
            "disruption.recovery.plan",
            request.train.train_id,
            at,
            lambda: summary,
            evidence={
                "decision": outcome.decision.value,
                "rollback_reason": outcome.rollback_reason or "",
                "alternatives_considered": len(outcome.evaluations),
            },
        )
        receipt.result["outcome"] = outcome
        return receipt
