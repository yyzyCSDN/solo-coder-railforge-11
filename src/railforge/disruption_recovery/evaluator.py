"""Combine safety, risk and customer-window scoring into evaluations."""
from __future__ import annotations

from . import risk as risk_mod
from . import safety as safety_mod
from . import scheduler
from .domain import (
    Evaluation,
    Finding,
    RecoveryOption,
    RecoveryRequest,
    Severity,
)


def lateness_by_commitment(request: RecoveryRequest, itinerary) -> dict[str, int]:
    """Minutes late per customer commitment (0 when met)."""
    out: dict[str, int] = {}
    for c in request.train.commitments:
        if c.destination != request.train.destination:
            out[c.car_id] = 0
            continue
        delta = (itinerary.arrive_at - c.deadline).total_seconds() / 60.0
        out[c.car_id] = max(0, int(round(delta)))
    return out


def broken_commitments(request: RecoveryRequest, lateness: dict[str, int]) -> tuple[str, ...]:
    limits = {c.car_id: c.max_tolerated_lateness_minutes
              for c in request.train.commitments}
    return tuple(sorted(
        car for car, minutes in lateness.items()
        if minutes > limits.get(car, 0)
    ))


def evaluate_option(request: RecoveryRequest, option: RecoveryOption) -> Evaluation:
    itinerary = scheduler.schedule(request, option)

    hard = safety_mod.evaluate(request, option, itinerary)
    if hard:
        return Evaluation(
            option=option,
            itinerary=itinerary,
            accepted=False,
            risk_score=1.0,
            max_lateness_minutes=0,
            broken_commitments=(),
            findings=tuple(hard),
        )

    breakdown = risk_mod.assess(request, option, itinerary)
    risk_findings = risk_mod.gate(request, breakdown)
    lateness = lateness_by_commitment(request, itinerary)
    broken = broken_commitments(request, lateness)

    findings: list[Finding] = list(risk_findings)
    commitment_findings: list[Finding] = []
    for car, minutes in sorted(lateness.items()):
        if minutes <= 0:
            continue
        if car in broken:
            commitment_findings.append(Finding(
                "COMMITMENT_BROKEN", Severity.COMMITMENT_BREACH,
                f"car {car} arrives {minutes} min after tolerated deadline"))
        else:
            findings.append(Finding(
                "COMMITMENT_LATE", Severity.INFO,
                f"car {car} arrives {minutes} min late (within tolerance)"))
    findings.extend(commitment_findings)

    # Accepted = safe, within risk appetite AND holding every commercial
    # commitment.  Lateness inside tolerance is allowed (it becomes a ranking
    # term); lateness beyond it is a refusal reason.
    accepted = not risk_findings and not commitment_findings

    return Evaluation(
        option=option,
        itinerary=itinerary,
        accepted=accepted,
        risk_score=breakdown.score,
        max_lateness_minutes=max(lateness.values(), default=0),
        broken_commitments=broken,
        findings=tuple(findings),
    )


def rank_key(evaluation: Evaluation):
    """Ordering for alternatives that are already safe, risk-controlled and
    holding every commercial commitment.

    Safety, risk appetite and commitments never appear here as trade-off
    terms: only accepted evaluations are ranked.  Among them we minimize
    lateness, then residual risk, then structural cost.
    """
    path = evaluation.option.path
    return (
        evaluation.max_lateness_minutes,
        round(evaluation.risk_score, 6),
        len(path),
        len(evaluation.option.power.all_ids),
        evaluation.option.option_id,
    )


def select(evaluations: list[Evaluation]) -> Evaluation | None:
    accepted = [e for e in evaluations if e.accepted]
    if not accepted:
        return None
    return min(accepted, key=rank_key)
