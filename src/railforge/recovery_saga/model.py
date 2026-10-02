"""Freight disruption recovery planning.

A recovery plan is only accepted when every hard safety gate passes for every
move.  Gates are committed constraints: punctuality is never bought by
relaxing a blocked route, the risk ceiling, the minimum brake percentage or
the traction requirement.  When no plan exists the structured rejection
reasons are returned so the orchestrator can roll the saga back and retain an
audit explanation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations

from railforge.braking.brake_percentage import Vehicle, meets as brakes_meet
from railforge.core.time import utc
from railforge.locomotive.tractive_effort import required_effort

# Rejection reasons, ordered from the hardest gate to commercial gates.
BLOCKED_SECTION = "blocked_section"
RISK_EXCEEDED = "risk_exceeded"
BRAKE_PERCENTAGE = "brake_percentage_below_minimum"
TRACTION_SHORTFALL = "traction_shortfall"
NO_LOCOMOTIVE = "no_locomotive_available"
DEADLINE_MISSED = "customer_deadline_missed"
LOCO_CONTENTION = "locomotive_resource_contention"
WAGON_CONTENTION = "wagon_resource_contention"
MALFORMED_INPUT = "malformed_input"
NO_ALTERNATIVE = "no_alternative_offered"


@dataclass(frozen=True)
class DisruptionInput:
    """The disruption handed to the recovery orchestrator."""

    disruption_id: str
    affected_paths: tuple[str, ...]
    blocked_sections: frozenset[str] = frozenset()
    since: datetime | None = None

    def __post_init__(self):
        if not self.disruption_id:
            raise ValueError("disruption id required")
        if not self.affected_paths:
            raise ValueError("at least one affected path required")
        if self.since is not None:
            utc(self.since)


@dataclass(frozen=True)
class AffectedMove:
    """A train move caught by the disruption with its customer commitment."""

    move_id: str
    train_id: str
    wagons: tuple[str, ...]
    tonnes: float
    gradient_permille: float
    curve_resistance_kn: float
    earliest_departure: datetime
    customer_deadline: datetime
    alternatives: tuple["AlternativePath", ...]

    def __post_init__(self):
        if not self.move_id or not self.train_id:
            raise ValueError("move and train ids required")
        if not self.wagons:
            raise ValueError("move carries no wagons")
        if self.tonnes <= 0 or self.curve_resistance_kn < 0:
            raise ValueError("invalid train mass")
        earliest = utc(self.earliest_departure)
        deadline = utc(self.customer_deadline)
        if deadline < earliest:
            raise ValueError("customer deadline before earliest departure")
        if not self.alternatives:
            raise ValueError("move has no alternative paths")


@dataclass(frozen=True)
class AlternativePath:
    """A candidate diversion for one move."""

    option_id: str
    path: tuple[str, ...]
    run_minutes: int
    cost: float
    risk: float
    required_capabilities: frozenset[str] = frozenset()

    def __post_init__(self):
        if not self.option_id or not self.path:
            raise ValueError("option id and path required")
        if self.run_minutes < 0 or self.cost < 0 or not (0.0 <= self.risk <= 1.0):
            raise ValueError("invalid alternative")


@dataclass(frozen=True)
class RecoveryLocomotive:
    """A locomotive that may be assigned to a recovery move."""

    loco_id: str
    continuous_kn: float
    adhesion_kn: float
    available: bool
    available_at: datetime
    location: str
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self):
        if not self.loco_id:
            raise ValueError("locomotive id required")
        if self.continuous_kn < 0 or self.adhesion_kn < 0:
            raise ValueError("invalid locomotive effort")
        utc(self.available_at)


@dataclass(frozen=True)
class SafetyPolicy:
    """Committed safety constraints.  These gates are never traded away."""

    max_risk: float = 0.25
    min_brake_percent: float = 85.0
    adhesion_factor: float = 1.0

    def __post_init__(self):
        if not (0.0 <= self.max_risk <= 1.0):
            raise ValueError("risk ceiling out of range")
        if self.min_brake_percent <= 0:
            raise ValueError("minimum brake percentage must be positive")
        if not (0.0 < self.adhesion_factor <= 1.0):
            raise ValueError("adhesion factor out of range")


@dataclass(frozen=True)
class Assignment:
    """The chosen alternative and consist for one move."""

    move_id: str
    option_id: str
    path: tuple[str, ...]
    locomotives: tuple[str, ...]
    projected_arrival: datetime
    cost: float
    risk: float
    on_time: bool


@dataclass(frozen=True)
class RecoveryPlan:
    """The complete, internally consistent recovery plan."""

    plan_id: str
    disruption_id: str
    assignments: tuple[Assignment, ...]
    wagons: tuple[str, ...]
    locomotives: tuple[str, ...]
    total_cost: float
    max_risk: float
    published_at: datetime | None = None


@dataclass(frozen=True)
class Rejection:
    """Auditable reason an alternative (or a move) could not be used."""

    move_id: str
    option_id: str | None
    reason: str
    detail: str = ""
    resource: str | None = None


class RecoveryInfeasible(Exception):
    """Raised when no risk-controlled, safety-compliant plan exists."""

    def __init__(self, rejections: tuple[Rejection, ...]):
        self.rejections = tuple(rejections)
        super().__init__("recovery infeasible: " + "; ".join(
            f"{r.move_id}/{r.option_id or '-'}={r.reason}" for r in self.rejections
        ))


def _effort_needed(move: AffectedMove) -> float:
    return required_effort(
        move.tonnes, move.gradient_permille, move.curve_resistance_kn
    )


def _available_locomotive(loco: RecoveryLocomotive, move: AffectedMove,
                          alternative: AlternativePath,
                          adhesion_factor: float) -> tuple[bool, str]:
    if not loco.available:
        return False, "not available"
    if loco.available_at > utc(move.earliest_departure):
        return False, "not yet serviceable"
    if not alternative.required_capabilities.issubset(loco.capabilities):
        return False, "missing capability"
    effort = min(loco.continuous_kn, loco.adhesion_kn * adhesion_factor)
    if effort <= 0:
        return False, "no usable tractive effort"
    return True, ""


def _eligible_locomotives(move, alternative, locomotives, adhesion_factor):
    """Locomotives that can individually serve this alternative."""
    eligible = []
    for loco in locomotives:
        ok, _ = _available_locomotive(loco, move, alternative, adhesion_factor)
        if ok:
            effort = min(loco.continuous_kn, loco.adhesion_kn * adhesion_factor)
            eligible.append((effort, loco))
    eligible.sort(key=lambda row: (-row[0], row[1].loco_id))
    return eligible


def _feasible_consists(move, alternative, locomotives, adhesion_factor, needed):
    """All inclusion-minimal feasible consists, strongest/shortest first."""
    eligible = _eligible_locomotives(move, alternative, locomotives,
                                     adhesion_factor)
    pool = [loco for _, loco in eligible]
    if not pool:
        return []
    effort_of = {
        loco.loco_id: min(loco.continuous_kn,
                          loco.adhesion_kn * adhesion_factor)
        for loco in pool
    }
    consists = []
    for size in range(1, len(pool) + 1):
        for group in combinations(pool, size):
            ids = tuple(l.loco_id for l in group)
            if sum(effort_of[i] for i in ids) + 1e-9 < needed:
                continue
            if not all(
                sum(effort_of[i] for i in ids if i != loco.loco_id) + 1e-9
                < needed
                for loco in group
            ):
                continue  # not minimal: a subset already meets the requirement
            consists.append(ids)
        if consists:
            return consists  # minimal size class, ordered strongest-first
    return []


def _screen_alternative(move, alternative, disruption, vehicles, locomotives,
                        policy):
    """Apply every committed safety gate; return None or a rejection."""
    blocked = [s for s in alternative.path if s in disruption.blocked_sections]
    if blocked:
        return Rejection(move.move_id, alternative.option_id,
                         BLOCKED_SECTION, "blocked: " + ",".join(sorted(blocked)),
                         resource=sorted(blocked)[0])
    if alternative.risk > policy.max_risk:
        return Rejection(move.move_id, alternative.option_id, RISK_EXCEEDED,
                         f"risk {alternative.risk:.3f} > {policy.max_risk:.3f}")
    move_vehicles = [vehicles[w] for w in move.wagons]
    if not brakes_meet(move_vehicles, move.gradient_permille,
                       policy.min_brake_percent):
        return Rejection(move.move_id, alternative.option_id, BRAKE_PERCENTAGE,
                         f"below {policy.min_brake_percent}%")
    needed = _effort_needed(move)
    consists = _feasible_consists(move, alternative, locomotives,
                                 policy.adhesion_factor, needed)
    if not consists:
        eligible = _eligible_locomotives(move, alternative, locomotives,
                                         policy.adhesion_factor)
        if not eligible:
            return Rejection(move.move_id, alternative.option_id, NO_LOCOMOTIVE,
                             "no serviceable locomotive with required capabilities")
        return Rejection(move.move_id, alternative.option_id, TRACTION_SHORTFALL,
                         f"available effort below required {needed:.1f} kN")
    arrival = utc(move.earliest_departure) + timedelta(
        minutes=alternative.run_minutes)
    if arrival > utc(move.customer_deadline):
        return Rejection(move.move_id, alternative.option_id, DEADLINE_MISSED,
                         f"arrival {arrival.isoformat()} after deadline "
                         f"{utc(move.customer_deadline).isoformat()}")
    return None


def _candidates(move, alternative, locomotives, policy):
    """One scored candidate per feasible consist for this alternative."""
    needed = _effort_needed(move)
    arrival = utc(move.earliest_departure) + timedelta(
        minutes=alternative.run_minutes)
    return [
        {
            "option": alternative,
            "locos": consist,
            "arrival": arrival,
            "key": (alternative.risk, alternative.cost,
                    alternative.run_minutes, alternative.option_id,
                    len(consist), consist),
        }
        for consist in _feasible_consists(move, alternative, locomotives,
                                          policy.adhesion_factor, needed)
    ]


def build_plan(disruption: DisruptionInput,
               moves: list[AffectedMove],
               vehicles: dict[str, Vehicle],
               locomotives: list[RecoveryLocomotive],
               policy: SafetyPolicy | None = None) -> RecoveryPlan:
    """Select the risk-controlled recovery plan or raise with audit reasons.

    Every alternative is screened against the committed safety gates.  Among
    the survivors the lowest-risk (then cheapest) combination is chosen with a
    deterministic backtracking search that keeps locomotive and wagon
    assignments disjoint across moves.
    """
    policy = policy or SafetyPolicy()
    if not moves:
        raise RecoveryInfeasible((Rejection("-", None, MALFORMED_INPUT,
                                            "no affected moves"),))
    move_ids = [m.move_id for m in moves]
    if len(set(move_ids)) != len(move_ids):
        raise RecoveryInfeasible((Rejection("-", None, MALFORMED_INPUT,
                                            "duplicate move id"),))
    unknown = sorted({w for m in moves for w in m.wagons if w not in vehicles})
    if unknown:
        raise RecoveryInfeasible((Rejection("-", None, MALFORMED_INPUT,
                                            "unknown wagons: " + ",".join(unknown),
                                            resource=unknown[0]),))
    loco_ids = [l.loco_id for l in locomotives]
    if len(set(loco_ids)) != len(loco_ids):
        raise RecoveryInfeasible((Rejection("-", None, MALFORMED_INPUT,
                                            "duplicate locomotive id"),))

    candidates: dict[str, list[dict]] = {}
    rejections: list[Rejection] = []
    for move in moves:
        local: list[dict] = []
        rejected_any = False
        for alternative in move.alternatives:
            rejection = _screen_alternative(
                move, alternative, disruption, vehicles, locomotives, policy
            )
            if rejection is not None:
                rejections.append(rejection)
                rejected_any = True
                continue
            local.extend(_candidates(move, alternative, locomotives, policy))
        if not local:
            if not rejected_any:
                rejections.append(Rejection(move.move_id, None, NO_ALTERNATIVE,
                                            "no alternative offered"))
            raise RecoveryInfeasible(tuple(rejections))
        local.sort(key=lambda c: c["key"])
        candidates[move.move_id] = local

    wagon_owners: dict[str, str] = {}
    for move in moves:
        for wagon in move.wagons:
            if wagon in wagon_owners:
                rejections.append(Rejection(move.move_id, None, WAGON_CONTENTION,
                                            f"wagon {wagon} also on {wagon_owners[wagon]}",
                                            resource=wagon))
            else:
                wagon_owners[wagon] = move.move_id
    if any(r.reason == WAGON_CONTENTION for r in rejections):
        raise RecoveryInfeasible(tuple(rejections))

    order = sorted(moves, key=lambda m: (
        len(candidates[m.move_id]), m.move_id))

    def search(index, used_locos, used_wagons):
        if index == len(order):
            return {}
        move = order[index]
        for cand in candidates[move.move_id]:
            locos = set(cand["locos"])
            wagons = set(move.wagons)
            if locos & used_locos or wagons & used_wagons:
                continue
            result = search(index + 1, used_locos | locos,
                            used_wagons | wagons)
            if result is not None:
                result[move.move_id] = cand
                return result
        return None

    solution = search(0, frozenset(), frozenset())
    if solution is None:
        contention = [
            Rejection(move.move_id, None, LOCO_CONTENTION,
                      "locomotives cannot be split across all moves")
            for move in moves
        ]
        raise RecoveryInfeasible(tuple(rejections + contention))

    assignments = []
    total_cost = 0.0
    max_risk = 0.0
    all_locos: set[str] = set()
    for move in moves:
        cand = solution[move.move_id]
        alternative = cand["option"]
        on_time = cand["arrival"] <= utc(move.customer_deadline)
        assignments.append(Assignment(
            move_id=move.move_id,
            option_id=alternative.option_id,
            path=alternative.path,
            locomotives=cand["locos"],
            projected_arrival=cand["arrival"],
            cost=alternative.cost,
            risk=alternative.risk,
            on_time=on_time,
        ))
        total_cost += alternative.cost
        max_risk = max(max_risk, alternative.risk)
        all_locos.update(cand["locos"])

    all_wagons = tuple(sorted({w for m in moves for w in m.wagons}))
    return RecoveryPlan(
        plan_id=f"{disruption.disruption_id}:recovery",
        disruption_id=disruption.disruption_id,
        assignments=tuple(assignments),
        wagons=all_wagons,
        locomotives=tuple(sorted(all_locos)),
        total_cost=total_cost,
        max_risk=max_risk,
    )
