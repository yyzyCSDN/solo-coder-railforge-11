"""Deterministic sample world used by the demo and the test suite.

Topology (all segments bidirectional)::

            B
          /   \\
         A-----E   (B-E blocked; A-C-E carries a hazmat prohibition)
         |\\   |
         | \\  |
         D--/
          \\
           C

* original routing A-B-E;
* A-C-E is shorter but hazmat prohibited and lightly rated;
* A-D-E is the controlled detour;
* unit L2 (certified only on A-D/D-E) is staged at A; L4 at C for the
  over-weight scenario; L3 is a helper staged at D.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .domain import (
    Blockage,
    Crew,
    CustomerCommitment,
    DisruptedTrain,
    Locomotive,
    Network,
    Railcar,
    RecoveryRequest,
    Segment,
)

U = timezone.utc
T0 = datetime(2026, 10, 2, 8, 0, tzinfo=U)


def build_network(blockage_hours: int = 5) -> Network:
    segments = (
        Segment("AB", "A", "B", 80, 60, max_gross_t=12000),
        Segment("BE", "B", "E", 120, 60, max_gross_t=12000, grade_permille=2),
        Segment("AD", "A", "D", 90, 60, max_gross_t=12000),
        Segment("DE", "D", "E", 130, 60, max_gross_t=12000, grade_permille=8),
        Segment("AC", "A", "C", 60, 60, max_gross_t=7000, hazmat_prohibited=True),
        Segment("CE", "C", "E", 60, 60, max_gross_t=7000, hazmat_prohibited=True),
        Segment("DC", "D", "C", 40, 40, max_gross_t=12000),
    )
    blockages = (
        Blockage(
            segment_id="BE",
            since=T0,
            estimated_clear=T0 + timedelta(hours=blockage_hours),
            clearance_uncertainty_minutes=480,
        ),
    )
    return Network(segments=segments, blockages=blockages)


def locomotives(scenario: str = "standard") -> tuple[tuple[Locomotive, ...], tuple[Locomotive, ...]]:
    """Return (on-train units, replacement pool)."""
    ab_be = frozenset({"AB", "BE"})
    ad_de = frozenset({"AD", "DE"})
    ac_ce = frozenset({"AC", "CE"})
    de_only = frozenset({"DE"})
    all_segs = frozenset({"AB", "BE", "AD", "DE", "AC", "CE", "DC"})

    if scenario == "standard":
        on_train = (
            Locomotive("L1", "A", 280.0, 900.0, ab_be, prep_minutes=0),
            Locomotive("L1B", "A", 280.0, 900.0, ab_be, prep_minutes=0),
        )
        pool = (
            # Controlled replacement for the graded detour.
            Locomotive("L2", "A", 380.0, 1200.0, ad_de, prep_minutes=10),
            # Helper staged at D, certified only on DE: required on the grade.
            Locomotive("L3", "D", 280.0, 900.0, de_only, prep_minutes=0),
            # Strong but uncertified on the detour; tempting and illegal.
            Locomotive("L4", "A", 260.0, 800.0, ac_ce | ab_be, prep_minutes=0),
        )
    else:  # heavy: no reroute path can carry the gross load
        on_train = (
            Locomotive("L1", "A", 280.0, 900.0, ab_be),
            Locomotive("L1B", "A", 280.0, 900.0, ab_be),
        )
        pool = (
            Locomotive("L2", "A", 380.0, 1200.0, ad_de, prep_minutes=10),
            Locomotive("L4", "C", 450.0, 1400.0, ac_ce, prep_minutes=0),
        )
    return on_train, pool


def cars(scenario: str = "standard", brake_hours: float = 24.0) -> tuple[Railcar, ...]:
    if scenario == "standard":
        rows = [Railcar(f"C{i:02d}", 1000.0, brake_valid_until=T0 + timedelta(hours=brake_hours))
                for i in range(1, 7)]
        rows[0] = Railcar("C01", 1000.0, hazmat=True,
                          brake_valid_until=T0 + timedelta(hours=brake_hours))
        return tuple(rows)
    rows = [Railcar(f"H{i:02d}", 1400.0, brake_valid_until=T0 + timedelta(hours=brake_hours))
            for i in range(1, 9)]
    rows[0] = Railcar("H01", 1400.0, hazmat=True,
                      brake_valid_until=T0 + timedelta(hours=brake_hours))
    return tuple(rows)


def build_train(scenario: str = "standard", brake_hours: float = 24.0,
                duty_used_hours: float = 8.5) -> DisruptedTrain:
    network = build_network()
    on_train, _ = locomotives(scenario)
    train_cars = cars(scenario, brake_hours)
    crew = Crew("K1", "A", duty_used_hours=duty_used_hours, qualified_hazmat=True)
    deadline = T0 + timedelta(hours=7)  # 15:00
    commitments = tuple(
        CustomerCommitment(c.car_id, "E", deadline, max_tolerated_lateness_minutes=120)
        for c in train_cars
    )
    return DisruptedTrain(
        train_id=f"7MS1-{scenario}",
        origin="A",
        destination="E",
        cars=train_cars,
        locomotives=on_train,
        crew=crew,
        original_path=("A", "B", "E"),
        ready_at=T0,
        blockage=network.blockage_on("BE"),
        commitments=commitments,
    )


def build_request(scenario: str = "standard", *, max_risk: float = 0.35,
                  brake_hours: float = 24.0, duty_used_hours: float = 8.5,
                  blockage_hours: int = 5) -> RecoveryRequest:
    network = build_network(blockage_hours=blockage_hours)
    train = build_train(scenario, brake_hours, duty_used_hours)
    if blockage_hours != 5:
        # Rebuild train so its blockage reference matches the network.
        train = DisruptedTrain(
            train_id=train.train_id, origin=train.origin, destination=train.destination,
            cars=train.cars, locomotives=train.locomotives, crew=train.crew,
            original_path=train.original_path, ready_at=train.ready_at,
            blockage=network.blockage_on("BE"), commitments=train.commitments,
        )
    _, pool = locomotives(scenario)
    relief = (Crew("K2", "D", duty_used_hours=2.0, qualified_hazmat=True),)
    return RecoveryRequest(
        request_id=f"REQ-{scenario.upper()}-001",
        train=train,
        network=network,
        loco_pool=pool,
        relief_crews=relief,
        crew_change_nodes=frozenset({"D"}),
        max_acceptable_risk=max_risk,
        initiated_at=T0,
    )
