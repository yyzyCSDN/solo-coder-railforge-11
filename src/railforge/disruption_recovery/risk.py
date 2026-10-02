"""Quantified risk assessment for safe alternatives.

Safety and risk are separate gates.  Safety findings reject outright; risk is a
bounded 0..1 score that must stay inside the corridor's published appetite
(``RecoveryRequest.max_acceptable_risk``).  Waiting on an uncertain blockage
clearance is deliberately expensive here: arriving *on time* by betting on an
early reopen is not the same as arriving safely.

Factor contributions (each bounded to [0, 1]):

* weather exposure        — severe weather along segments the run traverses;
* blockage uncertainty    — waiting time at risk vs. clearance uncertainty;
* traction margin         — small headroom above the required effort;
* brake reserve           — little time between arrival and inspection expiry;
* crew reserve            — little duty headroom left after the run.
"""
from __future__ import annotations

from dataclasses import dataclass

from .domain import Finding, Itinerary, RecoveryOption, RecoveryRequest, Severity

# Fixed, published weights.  They sum to 1.0 and are change-controlled out of
# band — orchestration callers cannot tune them to force a preferred option.
WEATHER_WEIGHT = 0.25
BLOCKAGE_WEIGHT = 0.30
TRACTION_WEIGHT = 0.20
BRAKE_WEIGHT = 0.15
CREW_WEIGHT = 0.10


@dataclass(frozen=True)
class RiskBreakdown:
    score: float
    weather: float
    blockage: float
    traction: float
    brake: float
    crew: float


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _weather_risk(request: RecoveryRequest, itinerary: Itinerary) -> float:
    """Maximum severe-weather exposure on any traversed segment.

    Severity lives on the segment in the caller's network model via an optional
    side-table keyed by segment id; absent data means zero measured exposure.
    """
    table = request.weather_severity or {}
    worst = 0.0
    for mv in itinerary.movements:
        worst = max(worst, float(table.get(mv.segment.segment_id, 0.0)))
    return _clamp(worst)


def _blockage_risk(request: RecoveryRequest, option: RecoveryOption,
                   itinerary: Itinerary) -> float:
    waiting = [mv for mv in itinerary.movements if mv.waiting_on_blockage]
    if not waiting:
        return 0.0
    blockage = request.network.blockage_on(waiting[0].segment.segment_id)
    if blockage is None or blockage.clearance_uncertainty_minutes <= 0:
        return 0.0
    # The whole uncertainty window is at risk; normalize against an 8-hour
    # planning horizon.  A full 8h of slack on a clearance => risk 1.0.
    return _clamp(blockage.clearance_uncertainty_minutes / (8.0 * 60.0))


def _traction_risk(request: RecoveryRequest, itinerary: Itinerary) -> float:
    locos = {u.loco_id: u for u in (*request.train.locomotives, *request.loco_pool)}
    worst = 0.0
    tonnes = request.train.gross_tonnes()
    for mv in itinerary.movements:
        seg = mv.segment
        required = tonnes * 9.80665 * (0.0018 + max(0.0, seg.grade_permille) / 1000.0) \
            + seg.curve_resistance_kn
        available = sum(
            min(locos[i].continuous_kn, locos[i].adhesion_kn * request.adhesion_factor)
            for i in mv.power_ids if i in locos
        )
        # Margin below 25% starts to add risk; below 10% would already be a
        # hard rejection in the safety gate.
        margin = (available - required) / required if required > 0 else 1.0
        worst = max(worst, _clamp((0.25 - margin) / 0.25))
    return worst


def _brake_risk(request: RecoveryRequest, itinerary: Itinerary) -> float:
    arrival = itinerary.arrive_at
    worst = 0.0
    for car in request.train.cars:
        if car.brake_valid_until is None:
            return 1.0
        reserve_minutes = (car.brake_valid_until - arrival).total_seconds() / 60.0
        if reserve_minutes <= 0:
            return 1.0
        # Less than 12h of inspection reserve is treated as elevated risk.
        worst = max(worst, _clamp(1.0 - reserve_minutes / (12.0 * 60.0)))
    return worst


def _crew_risk(request: RecoveryRequest, itinerary: Itinerary) -> float:
    crews = {c.crew_id: c for c in (request.train.crew, *request.relief_crews)}
    worst = 0.0
    for crew_id in {m.crew_id for m in itinerary.movements}:
        crew = crews.get(crew_id)
        if crew is None:
            return 1.0
        drive = sum((m.clear_at - m.enter_at).total_seconds() / 3600.0
                    for m in itinerary.movements if m.crew_id == crew_id)
        headroom = request.duty.max_duty_hours - crew.duty_used_hours - drive
        # Less than 2h duty headroom after the run is elevated risk.
        worst = max(worst, _clamp(1.0 - headroom / 2.0))
    return worst


def assess(request: RecoveryRequest, option: RecoveryOption,
           itinerary: Itinerary) -> RiskBreakdown:
    weather = _weather_risk(request, itinerary)
    blockage = _blockage_risk(request, option, itinerary)
    traction = _traction_risk(request, itinerary)
    brake = _brake_risk(request, itinerary)
    crew = _crew_risk(request, itinerary)
    score = (
        WEATHER_WEIGHT * weather
        + BLOCKAGE_WEIGHT * blockage
        + TRACTION_WEIGHT * traction
        + BRAKE_WEIGHT * brake
        + CREW_WEIGHT * crew
    )
    return RiskBreakdown(
        score=round(score, 6),
        weather=round(weather, 6),
        blockage=round(blockage, 6),
        traction=round(traction, 6),
        brake=round(brake, 6),
        crew=round(crew, 6),
    )


def gate(request: RecoveryRequest, breakdown: RiskBreakdown) -> list[Finding]:
    if breakdown.score <= request.max_acceptable_risk + 1e-9:
        return []
    return [Finding(
        "RISK_TOLERANCE_EXCEEDED",
        Severity.RISK_REJECTED,
        f"risk {breakdown.score:.3f} exceeds corridor appetite "
        f"{request.max_acceptable_risk:.3f}",
    )]
