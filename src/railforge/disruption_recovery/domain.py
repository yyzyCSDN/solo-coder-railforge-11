"""Domain model for freight disruption recovery.

All objects are immutable value types.  The model deliberately distinguishes
three kinds of facts:

* committed safety constraints (certifications, gross/axle limits, hazmat
  bans, brake validity, hours-of-service, traction sufficiency) — these gate
  every alternative and can never be traded away;
* measured risk factors (weather exposure, power margin, inspection reserve,
  blockage uncertainty) — these must stay inside a published tolerance;
* commercial facts (customer deadlines) — used only to order alternatives
  that are already safe and risk-controlled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """A bidirectional block between two junctions."""
    segment_id: str
    endpoint_a: str
    endpoint_b: str
    distance_km: float
    speed_limit_kmh: float
    max_gross_t: float
    grade_permille: float = 0.0
    curve_resistance_kn: float = 0.0
    hazmat_prohibited: bool = False

    def other_end(self, end: str) -> str:
        if end == self.endpoint_a:
            return self.endpoint_b
        if end == self.endpoint_b:
            return self.endpoint_a
        raise ValueError(f"{self.segment_id} does not touch {end}")

    def connects(self, a: str, b: str) -> bool:
        return {self.endpoint_a, self.endpoint_b} == {a, b}


@dataclass(frozen=True)
class Blockage:
    """A segment that is unavailable for an interval.

    ``clearance_uncertainty_minutes`` expresses how much later than
    ``estimated_clear`` the blockage might realistically be lifted.  Waiting on
    the blockage is therefore a risky bet rather than a guaranteed arrival.
    """
    segment_id: str
    since: datetime
    estimated_clear: datetime
    clearance_uncertainty_minutes: int = 0


@dataclass(frozen=True)
class Network:
    segments: tuple[Segment, ...] = ()
    blockages: tuple[Blockage, ...] = ()

    def segment(self, segment_id: str) -> Segment:
        for s in self.segments:
            if s.segment_id == segment_id:
                return s
        raise ValueError(f"unknown segment {segment_id}")

    def blockage_on(self, segment_id: str) -> Blockage | None:
        for b in self.blockages:
            if b.segment_id == segment_id:
                return b
        return None

    def neighbors(self, node: str) -> tuple[tuple[str, Segment], ...]:
        out = []
        for s in self.segments:
            if s.endpoint_a == node:
                out.append((s.endpoint_b, s))
            elif s.endpoint_b == node:
                out.append((s.endpoint_a, s))
        return tuple(out)

    def path_segments(self, path: tuple[str, ...]) -> tuple[Segment, ...]:
        seg = {(s.endpoint_a, s.endpoint_b): s for s in self.segments}
        out = []
        for a, b in zip(path, path[1:]):
            s = seg.get((a, b)) or seg.get((b, a))
            if s is None:
                raise ValueError(f"no segment between {a} and {b}")
            out.append(s)
        return tuple(out)


# ---------------------------------------------------------------------------
# Rolling stock, traction and crew
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Railcar:
    car_id: str
    gross_t: float
    hazmat: bool = False
    # Timestamp up to which the last full air-brake inspection remains valid.
    brake_valid_until: datetime | None = None
    axle_t: float | None = None


@dataclass(frozen=True)
class Locomotive:
    loco_id: str
    location: str
    continuous_kn: float
    adhesion_kn: float
    # Segments this unit (and its assigned engineer) is certified to work.
    certified_segments: frozenset[str] = frozenset()
    # Minimum lead time before a reassigned unit can move (fuel, crew, prep).
    prep_minutes: int = 0
    available: bool = True


@dataclass(frozen=True)
class Crew:
    crew_id: str
    location: str
    # Duty already accumulated before the recovery run starts.
    duty_used_hours: float = 0.0
    available: bool = True
    qualified_hazmat: bool = True


@dataclass(frozen=True)
class DutyLimits:
    max_duty_hours: float = 12.0
    max_drive_hours: float = 10.0
    max_span_hours: float = 12.0


# ---------------------------------------------------------------------------
# Train, customer commitments and the disruption request
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CustomerCommitment:
    car_id: str
    destination: str
    deadline: datetime
    # Lateness beyond which the commitment is considered broken, not merely late.
    max_tolerated_lateness_minutes: int = 60


@dataclass(frozen=True)
class DisruptedTrain:
    train_id: str
    origin: str
    destination: str
    cars: tuple[Railcar, ...]
    locomotives: tuple[Locomotive, ...]
    crew: Crew
    original_path: tuple[str, ...]
    ready_at: datetime
    blockage: Blockage
    commitments: tuple[CustomerCommitment, ...]

    def gross_tonnes(self) -> float:
        return sum(c.gross_t for c in self.cars)

    def has_hazmat(self) -> bool:
        return any(c.hazmat for c in self.cars)


@dataclass(frozen=True)
class RecoveryRequest:
    request_id: str
    train: DisruptedTrain
    network: Network
    # Locomotive pool that may be substituted or added as helpers (excluding the
    # units already on the train).
    loco_pool: tuple[Locomotive, ...] = ()
    # Relief crews available at nodes other than the origin.
    relief_crews: tuple[Crew, ...] = ()
    duty: DutyLimits = field(default_factory=DutyLimits)
    # Nodes at which a crew handoff is operationally possible.
    crew_change_nodes: frozenset[str] = frozenset()
    # Published, change-controlled risk appetite for this corridor.
    max_acceptable_risk: float = 0.35
    required_power_margin: float = 0.10
    adhesion_factor: float = 0.33
    dwell_minutes: int = 20
    crew_change_minutes: int = 30
    # Severe-weather exposure per segment id, normalized to [0.0, 1.0].
    weather_severity: dict[str, float] = field(default_factory=dict)
    initiated_at: datetime | None = None
    actor: str = "dispatch"


# ---------------------------------------------------------------------------
# Itinerary produced by the scheduling stage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Movement:
    segment: Segment
    enter_at: datetime
    clear_at: datetime
    power_ids: tuple[str, ...]
    crew_id: str
    waiting_on_blockage: bool = False
    pickup_loco: bool = False
    crew_changed: bool = False


@dataclass(frozen=True)
class Itinerary:
    path: tuple[str, ...]
    depart_at: datetime
    arrive_at: datetime
    movements: tuple[Movement, ...]
    power_ids: tuple[str, ...]
    crew_changes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Alternatives, evaluation and the published plan
# ---------------------------------------------------------------------------

class OptionKind(str, Enum):
    WAIT = "wait_on_blockage"
    REROUTE = "reroute"


@dataclass(frozen=True)
class PowerAssignment:
    """Power consist for the run.

    ``head`` units leave with the train; ``helpers`` are picked up at a node
    (currently modelled at the first node on the reroute path).
    """
    head: tuple[str, ...]
    helpers: tuple[str, ...] = ()
    helper_pickup_node: str | None = None

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.head + self.helpers


@dataclass(frozen=True)
class RecoveryOption:
    option_id: str
    kind: OptionKind
    path: tuple[str, ...]
    power: PowerAssignment
    crew_changes: tuple[str, ...]
    depart_at: datetime
    rationale: str = ""


class Severity(str, Enum):
    INFO = "info"
    HARD_VIOLATION = "hard_violation"
    RISK_REJECTED = "risk_rejected"
    # A commercial commitment (customer deadline beyond tolerated lateness).
    # Tracked separately from safety so the audit record can never blur a
    # refused safety waiver with a missed service promise.
    COMMITMENT_BREACH = "commitment_breach"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    segment_id: str | None = None


@dataclass(frozen=True)
class Evaluation:
    option: RecoveryOption
    itinerary: Itinerary | None
    accepted: bool
    risk_score: float
    max_lateness_minutes: int
    broken_commitments: tuple[str, ...]
    findings: tuple[Finding, ...]

    @property
    def hard_rejections(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.HARD_VIOLATION)

    @property
    def risk_findings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.RISK_REJECTED)

    @property
    def commitment_findings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings
                     if f.severity is Severity.COMMITMENT_BREACH)


@dataclass(frozen=True)
class RecoveryPlan:
    plan_id: str
    request_id: str
    train_id: str
    option_id: str
    kind: OptionKind
    path: tuple[str, ...]
    power_ids: tuple[str, ...]
    crew_changes: tuple[str, ...]
    depart_at: datetime
    arrive_at: datetime
    movements: tuple[Movement, ...]
    risk_score: float
    max_lateness_minutes: int
    published_at: datetime


# ---------------------------------------------------------------------------
# Decision codes (kept stable for audit/reporting)
# ---------------------------------------------------------------------------

class DecisionCode(str, Enum):
    PUBLISHED = "published"
    NO_CONTROLLED_ALTERNATIVE = "no_controlled_alternative"
    PUBLICATION_CONFLICT = "publication_conflict"
    PUBLICATION_FAILED = "publication_failed"
    ROLLED_BACK = "rolled_back"
    ESCALATED = "escalated"


class AuditCode(str, Enum):
    REQUEST_RECEIVED = "recovery.request.received"
    OPTION_GENERATED = "recovery.option.generated"
    OPTION_REJECTED_SAFETY = "recovery.option.rejected.safety"
    OPTION_REJECTED_RISK = "recovery.option.rejected.risk"
    OPTION_REJECTED_COMMITMENT = "recovery.option.rejected.commitment"
    OPTION_SCORED = "recovery.option.scored"
    OPTION_SELECTED = "recovery.option.selected"
    PLAN_COMMITTED = "recovery.plan.committed"
    PLAN_PUBLISH_FAILED = "recovery.plan.publish_failed"
    PLAN_ROLLED_BACK = "recovery.plan.rolled_back"
    RECOVERY_ABORTED = "recovery.aborted"
