"""Generate the universe of recovery alternatives.

Generation is permissive on purpose: a candidate may be unsafe, and the safety
stage is the single place that says so.  Generating unsafe alternatives is
important because their rejection (with the reason) is what proves that safety
was not silently traded away.

Alternatives are of two kinds:

* ``WAIT`` — keep the original path and wait for the blocked segment to clear;
* ``REROUTE`` — take an alternate simple path, possibly with substituted power
  and/or helper locomotives picked up en route, and optional crew handoffs at
  approved relief nodes.
"""
from __future__ import annotations

from datetime import timedelta

from .domain import (
    Network,
    OptionKind,
    PowerAssignment,
    RecoveryOption,
    RecoveryRequest,
)

MAX_SIMPLE_PATHS = 25
MAX_HELPERS = 2


def _simple_paths(network: Network, origin: str, destination: str) -> list[tuple[str, ...]]:
    """All simple origin->destination paths, avoiding freshly blocked segments.

    A segment whose blockage has not been lifted cannot serve a reroute — the
    reroute exists precisely to route around the obstruction.  The WAIT option
    handles the case of entering a segment after it reopens.
    """
    results: list[tuple[str, ...]] = []

    def walk(node: str, path: list[str]) -> None:
        if len(results) >= MAX_SIMPLE_PATHS:
            return
        if node == destination:
            results.append(tuple(path))
            return
        for nxt, segment in network.neighbors(node):
            if nxt in path:
                continue
            # Reroutes never enter still-blocked segments; they must detour.
            if network.blockage_on(segment.segment_id) is not None:
                continue
            walk(nxt, path + [nxt])

    walk(origin, [origin])
    return results


def _head_power_variants(request: RecoveryRequest, path: tuple[str, ...]) -> list[PowerAssignment]:
    """Power consists that depart with the train at the origin."""
    pool = {u.loco_id: u for u in (*request.train.locomotives, *request.loco_pool)}
    at_origin = [u for u in pool.values() if u.location == request.train.origin and u.available]
    current = tuple(u.loco_id for u in request.train.locomotives if u.available)

    variants: list[PowerAssignment] = []
    seen: set[tuple[str, ...]] = set()

    def add(ids: tuple[str, ...]) -> None:
        key = tuple(sorted(ids))
        if 1 <= len(ids) <= 3 and key not in seen:
            seen.add(key)
            variants.append(PowerAssignment(head=ids))

    add(current)
    # Single and paired units from everything staged at the origin.
    units = [u.loco_id for u in at_origin]
    for u in units:
        add((u,))
    for i, u in enumerate(units):
        for v in units[i + 1:]:
            add((u, v))
    return variants


def _with_helpers(request: RecoveryRequest, base: PowerAssignment,
                  path: tuple[str, ...]) -> list[PowerAssignment]:
    """Augment a head consist with helpers picked up at the first path node."""
    if len(path) < 3:
        return [base]
    pickup = path[1]
    pool = {u.loco_id: u for u in (*request.train.locomotives, *request.loco_pool)}
    staged = [u.loco_id for u in pool.values() if u.location == pickup and u.available]
    staged = [u for u in staged if u not in base.head]
    out = [base]
    for u in staged[:MAX_HELPERS]:
        out.append(PowerAssignment(head=base.head, helpers=(u,), helper_pickup_node=pickup))
    return out


def _crew_variants(request: RecoveryRequest, path: tuple[str, ...],
                   kind: OptionKind) -> list[tuple[tuple[str, ...], int]]:
    """Crew handoff options.

    Returns tuples of (nodes at which crew changes, departure delay minutes).
    The no-change baseline is always present first.
    """
    train = request.train
    variants: list[tuple[tuple[str, ...], int]] = [((), 0)]
    if not request.relief_crews:
        return variants
    relief_by_node: dict[str, int] = {}
    for crew in request.relief_crews:
        if crew.available and crew.location in request.crew_change_nodes and crew.location in path:
            relief_by_node[crew.location] = 0
    if kind is OptionKind.WAIT:
        # A handoff while waiting at the origin has no value; relief crews live
        # at intermediate nodes on reroutes.
        return variants
    nodes = tuple(n for n in path[1:-1] if n in relief_by_node)
    for node in nodes:
        # The handoff happens en route; the scheduler inserts its dwell at the
        # change node, so it does not delay departure from the origin.
        variants.append(((node,), 0))
    return variants


def _departure_delay(request: RecoveryRequest, power: PowerAssignment) -> int:
    pool = {u.loco_id: u for u in (*request.train.locomotives, *request.loco_pool)}
    current = {u.loco_id for u in request.train.locomotives}
    added = [u for u in power.head if u not in current]
    return max((pool[u].prep_minutes for u in added), default=0)


def generate(request: RecoveryRequest) -> list[RecoveryOption]:
    """Enumerate every structurally possible recovery alternative."""
    train = request.train
    network = request.network
    original = tuple(train.original_path)

    options: list[RecoveryOption] = []
    seq = 0

    def add(kind: OptionKind, path: tuple[str, ...], power: PowerAssignment,
            crew_changes: tuple[str, ...], delay_minutes: int, rationale: str) -> None:
        nonlocal seq
        seq += 1
        depart = train.ready_at
        if delay_minutes:
            depart = depart + timedelta(minutes=delay_minutes)
        kind_tag = "W" if kind is OptionKind.WAIT else "R"
        options.append(RecoveryOption(
            option_id=f"{request.request_id}-{kind_tag}{seq:02d}",
            kind=kind,
            path=path,
            power=power,
            crew_changes=crew_changes,
            depart_at=depart,
            rationale=rationale,
        ))

    # 1) Wait for the blockage to clear on the original routing (original power).
    current_power = PowerAssignment(head=tuple(
        u.loco_id for u in train.locomotives if u.available))
    add(OptionKind.WAIT, original, current_power, (), 0,
        "hold original routing until blockage clears")

    # 2) Every alternate simple path (the original one is excluded — that is WAIT).
    for path in _simple_paths(network, train.origin, train.destination):
        if path == original:
            continue
        crew_options = _crew_variants(request, path, OptionKind.REROUTE)
        for base_power in _head_power_variants(request, path):
            for power in _with_helpers(request, base_power, path):
                delay = _departure_delay(request, power)
                for crew_changes, crew_delay in crew_options:
                    add(OptionKind.REROUTE, path, power, crew_changes,
                        delay + crew_delay, "reroute around blockage")

    return options
