"""Hard safety gate — committed safety constraints.

Every finding produced here has severity ``HARD_VIOLATION`` and rejects the
alternative unconditionally.  Risk, cost and punctality are considered only
*after* this gate; the orchestrator has no code path that accepts a plan in
spite of a hard finding.

Enforced commitments:

* ``SAFETY_BLOCKAGE_ENTERED``      — entering a segment that has not reopened;
* ``SAFETY_GROSS_TONNAGE``         — train gross above a segment rating;
* ``SAFETY_HAZMAT_PROHIBITED``     — placarded cars through a hazmat ban;
* ``SAFETY_LOCO_CERTIFICATION``    — an uncertified unit/engineer on a segment;
* ``SAFETY_LOCO_AVAILABILITY``     — unit not staged where the run needs it;
* ``SAFETY_TRACTION``              — insufficient continuous tractive effort,
                                     including the configured power margin;
* ``SAFETY_BRAKE_INSPECTION``      — air-brake inspection expired en route;
* ``SAFETY_HOURS_OF_SERVICE``      — crew beyond duty/drive/span limits;
* ``SAFETY_HAZMAT_QUALIFICATION``  — relief crew not hazmat qualified.
"""
from __future__ import annotations

from .domain import (
    Crew,
    Finding,
    Itinerary,
    Locomotive,
    RecoveryOption,
    RecoveryRequest,
    Severity,
)


def _locos(request: RecoveryRequest) -> dict[str, Locomotive]:
    return {u.loco_id: u for u in (*request.train.locomotives, *request.loco_pool)}


def _crew(request: RecoveryRequest) -> dict[str, Crew]:
    return {c.crew_id: c for c in (request.train.crew, *request.relief_crews)}


def _segment_effort(request: RecoveryRequest, segment) -> float:
    tonnes = request.train.gross_tonnes()
    rolling = tonnes * 9.80665 * 0.0018
    grade = tonnes * 9.80665 * (segment.grade_permille / 1000.0)
    return rolling + max(0.0, grade) + segment.curve_resistance_kn


def _available_effort(units: list[Locomotive], adhesion_factor: float) -> float:
    return sum(min(u.continuous_kn, u.adhesion_kn * adhesion_factor) for u in units)


def evaluate(request: RecoveryRequest, option: RecoveryOption,
             itinerary: Itinerary) -> list[Finding]:
    findings: list[Finding] = []
    locos = _locos(request)
    crews = _crew(request)
    network = request.network
    train = request.train

    def reject(code: str, message: str, segment_id: str | None = None) -> None:
        findings.append(Finding(code, Severity.HARD_VIOLATION, message, segment_id))

    # --- Power consist references real, usable units ----------------------
    for lid in option.power.all_ids:
        if lid not in locos:
            reject("SAFETY_LOCO_UNKNOWN", f"locomotive {lid} is not in the active roster", None)
            return findings  # cannot meaningfully run further checks
        if not locos[lid].available:
            reject("SAFETY_LOCO_AVAILABILITY",
                   f"locomotive {lid} is not available for service", None)

    seen_power: set[str] = set()

    for mv in itinerary.movements:
        seg = mv.segment

        # --- Blockage: no movement into a segment before it reopens --------
        blockage = network.blockage_on(seg.segment_id)
        if blockage is not None and mv.enter_at < blockage.estimated_clear:
            reject("SAFETY_BLOCKAGE_ENTERED",
                   f"enters {seg.segment_id} at {mv.enter_at.isoformat()} before estimated "
                   f"clearance {blockage.estimated_clear.isoformat()}", seg.segment_id)

        # --- Gross tonnage rating ------------------------------------------
        if train.gross_tonnes() > seg.max_gross_t:
            reject("SAFETY_GROSS_TONNAGE",
                   f"gross {train.gross_tonnes():.0f}t exceeds {seg.segment_id} rating "
                   f"{seg.max_gross_t:.0f}t", seg.segment_id)

        # --- Hazmat prohibition --------------------------------------------
        if train.has_hazmat() and seg.hazmat_prohibited:
            reject("SAFETY_HAZMAT_PROHIBITED",
                   f"placarded cars routed through hazmat-prohibited {seg.segment_id}",
                   seg.segment_id)

        # --- Per-unit certifications and staging ----------------------------
        for lid in mv.power_ids:
            unit = locos.get(lid)
            if unit is None:
                continue
            if seg.segment_id not in unit.certified_segments:
                reject("SAFETY_LOCO_CERTIFICATION",
                       f"locomotive {lid} not certified on {seg.segment_id}", seg.segment_id)
            # A unit must actually be reachable when first used: head units at
            # the origin, helpers at their pickup node.
            if lid not in seen_power:
                if lid in option.power.helpers:
                    expected_node = option.power.helper_pickup_node
                else:
                    expected_node = train.origin
                if unit.location != expected_node:
                    reject("SAFETY_LOCO_AVAILABILITY",
                           f"locomotive {lid} staged at {unit.location}, needed at {expected_node}",
                           seg.segment_id)
        seen_power.update(mv.power_ids)

        # --- Traction with the configured safety margin --------------------
        units = [locos[i] for i in mv.power_ids if i in locos]
        required = _segment_effort(request, seg) * (1.0 + request.required_power_margin)
        available = _available_effort(units, request.adhesion_factor)
        if available + 1e-9 < required:
            reject("SAFETY_TRACTION",
                   f"available effort {available:.1f}kN below required {required:.1f}kN "
                   f"(incl. {request.required_power_margin:.0%} margin) on {seg.segment_id}",
                   seg.segment_id)

    # --- Brake inspection validity through the whole run -------------------
    arrival = itinerary.arrive_at
    for car in train.cars:
        if car.brake_valid_until is None:
            reject("SAFETY_BRAKE_INSPECTION",
                   f"car {car.car_id} has no valid air-brake inspection record", None)
        elif car.brake_valid_until < arrival:
            reject("SAFETY_BRAKE_INSPECTION",
                   f"car {car.car_id} brake inspection expires "
                   f"{car.brake_valid_until.isoformat()} before arrival {arrival.isoformat()}",
                   None)

    # --- Hours of service for every crew that works part of the run --------
    duty = request.duty
    for crew_id in {m.crew_id for m in itinerary.movements}:
        crew = crews.get(crew_id)
        if crew is None:
            reject("SAFETY_HOURS_OF_SERVICE", f"unknown crew {crew_id}", None)
            continue
        if not crew.available:
            reject("SAFETY_HOURS_OF_SERVICE", f"crew {crew_id} is not available", None)
        # Drive time contributed by this crew on the recovery run.
        drive_hours = 0.0
        first = None
        last = None
        for mv in itinerary.movements:
            if mv.crew_id != crew_id:
                continue
            drive_hours += (mv.clear_at - mv.enter_at).total_seconds() / 3600.0
            first = mv.enter_at if first is None else min(first, mv.enter_at)
            last = max(last or mv.clear_at, mv.clear_at)
        span_hours = (last - first).total_seconds() / 3600.0 if first and last else 0.0
        duty_total = crew.duty_used_hours + drive_hours
        if drive_hours > duty.max_drive_hours + 1e-9:
            reject("SAFETY_HOURS_OF_SERVICE",
                   f"crew {crew_id} drives {drive_hours:.2f}h > {duty.max_drive_hours}h limit", None)
        if duty_total > duty.max_duty_hours + 1e-9:
            reject("SAFETY_HOURS_OF_SERVICE",
                   f"crew {crew_id} total duty {duty_total:.2f}h > {duty.max_duty_hours}h limit",
                   None)
        if span_hours > duty.max_span_hours + 1e-9:
            reject("SAFETY_HOURS_OF_SERVICE",
                   f"crew {crew_id} span {span_hours:.2f}h > {duty.max_span_hours}h limit", None)
        if train.has_hazmat() and not crew.qualified_hazmat:
            reject("SAFETY_HAZMAT_QUALIFICATION",
                   f"crew {crew_id} is not hazmat qualified for a placarded train", None)

    return findings
