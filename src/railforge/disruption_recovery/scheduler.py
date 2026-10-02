"""Turn a structural alternative into a timed itinerary.

Scheduling is purely physical/operational; it never decides whether a run is
*allowed*.  It reports where the train would wait, where power is picked up and
when each car would reach the destination.  Safety policy runs afterwards in
``safety.py``.
"""
from __future__ import annotations

from datetime import timedelta

from .domain import (
    Itinerary,
    Locomotive,
    Movement,
    RecoveryOption,
    RecoveryRequest,
)


def _loco_map(request: RecoveryRequest) -> dict[str, Locomotive]:
    return {u.loco_id: u for u in (*request.train.locomotives, *request.loco_pool)}


def schedule(request: RecoveryRequest, option: RecoveryOption) -> Itinerary:
    train = request.train
    network = request.network
    locos = _loco_map(request)
    segments = network.path_segments(option.path)

    cursor = option.depart_at
    current_crew = train.crew.crew_id
    movements: list[Movement] = []
    crew_changes: list[str] = []

    active_power = tuple(option.power.head)
    pickup_node = option.power.helper_pickup_node

    for i, segment in enumerate(segments):
        node_from = option.path[i]
        waiting = False
        pickup = False

        # Helper power joins at the first intermediate node where staged.
        if pickup_node is not None and node_from == pickup_node and option.power.helpers:
            active_power = active_power + tuple(option.power.helpers)
            pickup = True
            cursor = cursor + timedelta(minutes=request.dwell_minutes)

        # Approved crew handoff at this node: dwell + the relief crew takes over.
        if node_from in option.crew_changes:
            relief = next((c for c in request.relief_crews if c.location == node_from), None)
            if relief is not None:
                cursor = cursor + timedelta(minutes=request.crew_change_minutes)
                current_crew = relief.crew_id
                crew_changes.append(node_from)

        enter = cursor
        blockage = network.blockage_on(segment.segment_id)
        if blockage is not None:
            # Only the WAIT option may legally rely on a future clearance; the
            # scheduler still computes the wait so the risk stage can price the
            # uncertainty around the estimate.
            if enter < blockage.estimated_clear:
                enter = blockage.estimated_clear
                waiting = True

        travel_minutes = segment.distance_km / segment.speed_limit_kmh * 60.0
        clear = enter + timedelta(minutes=travel_minutes)
        movements.append(Movement(
            segment=segment,
            enter_at=enter,
            clear_at=clear,
            power_ids=active_power,
            crew_id=current_crew,
            waiting_on_blockage=waiting,
            pickup_loco=pickup,
            crew_changed=node_from in option.crew_changes,
        ))
        cursor = clear
        # Standard dwell at every intermediate node; pickup and handoff events
        # each add their own time on top of it.
        if i < len(segments) - 1:
            cursor = cursor + timedelta(minutes=request.dwell_minutes)

    return Itinerary(
        path=option.path,
        depart_at=option.depart_at,
        arrive_at=cursor,
        movements=tuple(movements),
        power_ids=option.power.all_ids,
        crew_changes=tuple(crew_changes),
    )
