import unittest
from datetime import datetime, timezone, timedelta

from railforge.api import RailForgeService
from railforge.audit.trail import Trail
from railforge.braking.brake_percentage import Vehicle
from railforge.events.stream import EventStream
from railforge.recovery_saga import (
    AffectedMove,
    AlternativePath,
    DisruptionInput,
    RecoveryAborted,
    RecoveryInfeasible,
    RecoveryLocomotive,
    RecoveryOrchestrator,
    ResourceLedger,
    SafetyPolicy,
    build_plan,
)
from railforge.recovery_saga.orchestrator import (
    LOCOMOTIVE,
    NO_SAFE_PLAN,
    PUBLICATION_FAILED,
    RESOURCE_CONTENTION,
    RECOVERY_TOPIC,
    WAGON,
)
from railforge.recovery_saga.model import (
    BLOCKED_SECTION,
    BRAKE_PERCENTAGE,
    DEADLINE_MISSED,
    RISK_EXCEEDED,
    TRACTION_SHORTFALL,
)
from railforge.storage.ops_store import VersionedStore

U = timezone.utc
AT = datetime(2026, 6, 1, 12, tzinfo=U)


def vehicles():
    # brake percentage = 90/100*100 = 90 >= 85
    return {
        "W1": Vehicle("W1", 60.0, 55.0),
        "W2": Vehicle("W2", 40.0, 35.0),
        "W3": Vehicle("W3", 50.0, 46.0),
    }


def locomotives():
    depart = datetime(2026, 6, 1, 11, tzinfo=U)
    return [
        RecoveryLocomotive("L1", 300.0, 300.0, True, depart, "Y",
                           frozenset({"dpu"})),
        RecoveryLocomotive("L2", 120.0, 120.0, True, depart, "Y"),
        RecoveryLocomotive("L3", 120.0, 120.0, False, depart, "Y"),
        RecoveryLocomotive("L4", 120.0, 120.0, True,
                           datetime(2026, 6, 1, 18, tzinfo=U), "Y"),
    ]


def disruption(blocked=("SEC-MAIN",)):
    return DisruptionInput("D1", ("P-MAIN",), frozenset(blocked), AT)


def move(option_id="DIV-A", run=180, risk=0.10, cost=5000.0,
         caps=frozenset(), deadline_hours=6, wagons=("W1", "W2"),
         gradient=0.0, curve=0.0, tonnes=100.0, path=("SEC-DIV",)):
    return AffectedMove(
        move_id="M1", train_id="T1", wagons=wagons, tonnes=tonnes,
        gradient_permille=gradient, curve_resistance_kn=curve,
        earliest_departure=AT,
        customer_deadline=AT + timedelta(hours=deadline_hours),
        alternatives=(AlternativePath(option_id, path, run, cost, risk, caps),))


def alternative(option_id, run=180, risk=0.10, cost=5000.0,
                caps=frozenset(), path=("SEC-DIV",)):
    return AlternativePath(option_id, path, run, cost, risk, caps)


class ModelTests(unittest.TestCase):
    def test_picks_lowest_risk_safe_alternative(self):
        m = AffectedMove(
            "M1", "T1", ("W1", "W2"), 100.0, 0.0, 0.0, AT,
            AT + timedelta(hours=6),
            (alternative("FAST", run=120, risk=0.20, cost=9000.0),
             alternative("SAFE", run=200, risk=0.05, cost=4000.0),
             alternative("CHEAP", run=240, risk=0.24, cost=1000.0)))
        plan = build_plan(disruption(), [m], vehicles(), locomotives())
        self.assertEqual(plan.assignments[0].option_id, "SAFE")
        self.assertTrue(plan.assignments[0].on_time)
        self.assertAlmostEqual(plan.max_risk, 0.05)
        self.assertEqual(plan.plan_id, "D1:recovery")

    def test_blocked_section_is_a_hard_gate(self):
        m = move(path=("SEC-MAIN", "SEC-X"))  # SEC-MAIN blocked
        with self.assertRaises(RecoveryInfeasible) as ctx:
            build_plan(disruption(), [m], vehicles(), locomotives())
        reasons = {r.reason for r in ctx.exception.rejections}
        self.assertEqual(reasons, {BLOCKED_SECTION})

    def test_risk_ceiling_is_never_relaxed_for_ontime(self):
        # On-time only via the risky path; the safe path misses the deadline.
        m = AffectedMove(
            "M1", "T1", ("W1", "W2"), 100.0, 0.0, 0.0, AT,
            AT + timedelta(hours=3),
            (alternative("RISKY", run=60, risk=0.90, cost=1000.0),
             alternative("SAFE-LATE", run=300, risk=0.05, cost=1000.0)))
        with self.assertRaises(RecoveryInfeasible) as ctx:
            build_plan(disruption(), [m], vehicles(), locomotives(),
                       SafetyPolicy(max_risk=0.25, min_brake_percent=85.0))
        reasons = {r.reason for r in ctx.exception.rejections}
        self.assertEqual(reasons, {RISK_EXCEEDED, DEADLINE_MISSED})

    def test_brake_percentage_gate_rejects_unsafe_consist(self):
        weak = {"W1": Vehicle("W1", 100.0, 80.0)}  # 80% < 85%
        m = move(wagons=("W1",))
        with self.assertRaises(RecoveryInfeasible) as ctx:
            build_plan(disruption(), [m], weak, locomotives(),
                       SafetyPolicy(min_brake_percent=85.0))
        self.assertEqual(ctx.exception.rejections[0].reason, BRAKE_PERCENTAGE)

    def test_traction_shortfall_rejects_even_if_on_time(self):
        # 1000 t needs ~17.7 kN rolling alone; the 120 kN units suffice only on
        # the flat — inflate the consist mass so the single strong unit fails.
        m = AffectedMove(
            "M1", "T1", ("W1", "W2"), 20000.0, 10.0, 100.0, AT,
            AT + timedelta(hours=12), (alternative("DIV-A", run=120),))
        with self.assertRaises(RecoveryInfeasible) as ctx:
            build_plan(disruption(), [m], vehicles(), locomotives())
        reasons = {r.reason for r in ctx.exception.rejections}
        self.assertIn(TRACTION_SHORTFALL, reasons)

    def test_capability_required_and_satisfied(self):
        m = move(caps=frozenset({"dpu"}))
        plan = build_plan(disruption(), [m], vehicles(), locomotives())
        self.assertEqual(plan.assignments[0].locomotives, ("L1",))

    def test_unavailable_or_not_yet_serviceable_locomotives_skipped(self):
        m = move(caps=frozenset({"dpu"}))
        # Only L1 has dpu; make the same capability scarce via a pool whose
        # capable unit is still under maintenance.
        pool = [RecoveryLocomotive("L9", 300.0, 300.0, True,
                                   datetime(2026, 6, 1, 18, tzinfo=U), "Y",
                                   frozenset({"dpu"}))]
        with self.assertRaises(RecoveryInfeasible) as ctx:
            build_plan(disruption(), [m], vehicles(), pool)
        self.assertEqual(ctx.exception.rejections[0].reason, "no_locomotive_available")

    def test_locomotives_cannot_be_split_across_moves(self):
        m1 = AffectedMove(
            "M1", "T1", ("W1",), 100.0, 0.0, 0.0, AT,
            AT + timedelta(hours=6), (alternative("A1"),))
        m2 = AffectedMove(
            "M2", "T2", ("W3",), 100.0, 0.0, 0.0, AT,
            AT + timedelta(hours=6), (alternative("A2"),))
        with self.assertRaises(RecoveryInfeasible) as ctx:
            build_plan(disruption(), [m1, m2], vehicles(),
                      [RecoveryLocomotive("L1", 300.0, 300.0, True, AT, "Y")])
        reasons = {r.reason for r in ctx.exception.rejections}
        self.assertIn("locomotive_resource_contention", reasons)

    def test_wagon_on_two_moves_is_a_contention(self):
        m1 = AffectedMove("M1", "T1", ("W1",), 100.0, 0.0, 0.0, AT,
                          AT + timedelta(hours=6), (alternative("A1"),))
        m2 = AffectedMove("M2", "T2", ("W1",), 100.0, 0.0, 0.0, AT,
                          AT + timedelta(hours=6), (alternative("A2"),))
        with self.assertRaises(RecoveryInfeasible) as ctx:
            build_plan(disruption(), [m1, m2], vehicles(), locomotives())
        self.assertEqual(ctx.exception.rejections[0].reason,
                         "wagon_resource_contention")

    def test_multi_move_plan_uses_disjoint_consists(self):
        m1 = AffectedMove("M1", "T1", ("W1",), 100.0, 0.0, 0.0, AT,
                          AT + timedelta(hours=6), (alternative("A1"),))
        m2 = AffectedMove("M2", "T2", ("W3",), 100.0, 0.0, 0.0, AT,
                          AT + timedelta(hours=6), (alternative("A2"),))
        plan = build_plan(disruption(), [m1, m2], vehicles(), locomotives())
        used = [a.locomotives for a in plan.assignments]
        self.assertEqual(set(used[0]) & set(used[1]), set())


class LedgerTests(unittest.TestCase):
    def test_stage_idempotent_within_operation_but_contended_across(self):
        ledger = ResourceLedger()
        ledger.stage("op-1", WAGON, "W1", "M1")
        ledger.stage("op-1", WAGON, "W1", "M1")  # replay within op
        self.assertEqual(ledger.held_by(WAGON, "W1"), "op-1")
        with self.assertRaises(Exception):
            ledger.stage("op-2", WAGON, "W1", "M2")

    def test_rollback_releases_everything(self):
        ledger = ResourceLedger()
        ledger.stage("op-1", WAGON, "W1", "M1")
        ledger.stage("op-1", LOCOMOTIVE, "L1", "M1")
        released = ledger.rollback("op-1")
        self.assertEqual(len(released), 2)
        self.assertIsNone(ledger.held_by(WAGON, "W1"))
        self.assertIsNone(ledger.held_by(LOCOMOTIVE, "L1"))
        # another operation can now take the resources
        ledger.stage("op-2", WAGON, "W1", "M2")
        self.assertEqual(ledger.held_by(WAGON, "W1"), "op-2")

    def test_commit_retains_leases(self):
        ledger = ResourceLedger()
        ledger.stage("op-1", WAGON, "W1", "M1")
        ledger.commit("op-1")
        self.assertEqual(ledger.held_by(WAGON, "W1"), "op-1")
        self.assertEqual(ledger.rollback("op-1"), ())


class FailingControl:
    """Control plane whose publication always fails after staging."""

    def __init__(self):
        self.stream = EventStream()
        self.trail = Trail()
        self.store = VersionedStore()

    def run(self, *args, **kwargs):
        raise RuntimeError("store outage")


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.service = RailForgeService()
        self.orchestrator = self.service.recovery

    def _publish(self, op="op-1", m=None, **kwargs):
        return self.orchestrator.publish(
            op, disruption(), [m or move()], vehicles(), locomotives(), AT,
            **kwargs)

    def test_safe_plan_published_atomically(self):
        receipt = self._publish()
        self.assertEqual(receipt.event.topic, RECOVERY_TOPIC)
        self.assertEqual(receipt.event.payload["status"], "committed")
        self.assertEqual(receipt.event.payload["control_path"], "atomic")
        self.assertEqual(receipt.event.payload["safety_hold"], True)
        plan = receipt.result
        self.assertEqual(plan.assignments[0].option_id, "DIV-A")
        self.assertEqual(plan.published_at, AT)
        self.assertTrue(self.service.workflow().stream.verify_chain())
        self.assertTrue(self.service.workflow().trail.verify())
        store_row = self.service.workflow().store.read("D1")
        self.assertEqual(store_row.version, 1)
        self.assertEqual(store_row.value["plan_id"], "D1:recovery")

    def test_replay_returns_same_receipt_and_does_not_republish(self):
        first = self._publish()
        second = self._publish()
        self.assertIs(first, second)
        self.assertEqual(self.service.workflow().stream.head, 1)

    def test_infeasible_plan_rolls_back_with_audit_reasons(self):
        m = move(path=("SEC-MAIN",))  # blocked
        with self.assertRaises(RecoveryAborted) as ctx:
            self._publish(op="op-bad", m=m)
        record = ctx.exception.record
        self.assertEqual(record.reason_code, NO_SAFE_PLAN)
        self.assertEqual(record.rejections[0].reason, BLOCKED_SECTION)
        # Nothing published, nothing staged, but the reason is audited.
        self.assertIsNone(self.service.workflow().store.read("D1"))
        self.assertEqual(self.service.workflow().stream.head, 0)
        self.assertTrue(self.service.workflow().trail.verify())
        trail_row = self.service.workflow().trail.record(record.audit_index)
        self.assertEqual(trail_row.action, "recovery.plan.rolled_back")
        self.assertEqual(trail_row.subject, "D1")
        self.assertEqual(trail_row.digest, record.audit_digest)
        # Retained in the rollback journal keyed by operation id.
        retained = self.orchestrator.rollbacks["op-bad"]
        self.assertIs(retained, record)
        self.assertEqual(self.orchestrator.journal[0].status, "rolled_back")

    def test_replay_of_failed_operation_raises_again_without_new_audit(self):
        m = move(risk=0.99)
        for _ in range(2):
            with self.assertRaises(RecoveryAborted) as ctx:
                self._publish(op="op-fail", m=m)
            self.assertEqual(ctx.exception.reason_code, NO_SAFE_PLAN)
        # one rollback audit row only
        rows = [r for r in self.service.workflow().trail._rows]
        self.assertEqual(len(rows), 1)

    def test_unsafe_punctuality_tradeoff_is_refused(self):
        # The only on-time move violates the risk ceiling: no local on-time
        # recovery may be force-published.
        m = AffectedMove(
            "M1", "T1", ("W1", "W2"), 100.0, 0.0, 0.0, AT,
            AT + timedelta(hours=2),
            (alternative("RISKY", run=60, risk=0.99, cost=100.0),))
        with self.assertRaises(RecoveryAborted) as ctx:
            self._publish(op="op-unsafe", m=m)
        reasons = {r.reason for r in ctx.exception.record.rejections}
        self.assertIn(RISK_EXCEEDED, reasons)
        self.assertIsNone(self.service.workflow().store.read("D1"))

    def test_resource_contention_between_sagas_rolls_back_second(self):
        first = self.orchestrator.publish(
            "op-1", disruption(), [move()], vehicles(), locomotives(), AT)
        self.assertEqual(first.result.locomotives, ("L1",))
        with self.assertRaises(RecoveryAborted) as ctx:
            self.orchestrator.publish(
                "op-2", disruption(blocked=("OTHER",)),
                [AffectedMove("M1", "T9", ("W3",), 100.0, 0.0, 0.0, AT,
                              AT + timedelta(hours=6),
                              (alternative("DIV-A"),))],
                vehicles(), locomotives(), AT)
        record = ctx.exception.record
        self.assertEqual(record.reason_code, RESOURCE_CONTENTION)
        self.assertEqual(record.rejections[0].resource, "L1")
        # The loser fully releases its staged wagons.
        self.assertIsNone(self.orchestrator.ledger.held_by(WAGON, "W3"))
        # The winner keeps its resources and remains the single publication.
        self.assertEqual(self.orchestrator.ledger.held_by(LOCOMOTIVE, "L1"),
                         "op-1")
        self.assertEqual(self.service.workflow().stream.head, 1)

    def test_publication_failure_rolls_back_staged_resources(self):
        orchestrator = RecoveryOrchestrator(FailingControl())
        with self.assertRaises(RecoveryAborted) as ctx:
            orchestrator.publish("op-x", disruption(), [move()], vehicles(),
                                 locomotives(), AT)
        record = ctx.exception.record
        self.assertEqual(record.reason_code, PUBLICATION_FAILED)
        self.assertEqual(len(record.released), 3)  # W1, W2, L1
        kinds = sorted((l.resource_kind for l in record.released))
        self.assertEqual(kinds, [LOCOMOTIVE, WAGON, WAGON])
        self.assertIsNone(orchestrator.ledger.held_by(WAGON, "W1"))
        self.assertIsNone(orchestrator.ledger.held_by(LOCOMOTIVE, "L1"))
        # The failed publication is still audited with its reason.
        self.assertTrue(orchestrator.trail.verify())
        self.assertEqual(orchestrator.journal[0].status, "rolled_back")

        # After rollback a healthy control plane can publish the same move.
        healthy = RecoveryOrchestrator(ledger=orchestrator.ledger)
        receipt = healthy.publish("op-y", disruption(), [move()], vehicles(),
                                  locomotives(), AT)
        self.assertEqual(receipt.event.payload["status"], "committed")

    def test_journal_records_both_outcomes_in_order(self):
        with self.assertRaises(RecoveryAborted):
            self._publish(op="op-a", m=move(path=("SEC-MAIN",)))
        self._publish(op="op-b")
        statuses = [e.status for e in self.orchestrator.journal]
        self.assertEqual(statuses, ["rolled_back", "committed"])
        # each journal entry binds to a distinct audit record
        indexes = {e.audit_index for e in self.orchestrator.journal}
        self.assertEqual(len(indexes), 2)

    def test_rolled_back_plan_carries_full_structured_reasons(self):
        m = AffectedMove(
            "M1", "T1", ("W1", "W2"), 100.0, 0.0, 0.0, AT,
            AT + timedelta(hours=6),
            (AlternativePath("BAD", ("SEC-MAIN",), 10, 1.0, 0.99),
             AlternativePath("RISKY", ("SEC-DIV",), 10, 1.0, 0.80)))
        with self.assertRaises(RecoveryAborted) as ctx:
            self._publish(op="op-reasons", m=m)
        rejections = ctx.exception.record.rejections
        by_option = {r.option_id: r.reason for r in rejections}
        self.assertEqual(by_option["BAD"], BLOCKED_SECTION)
        self.assertEqual(by_option["RISKY"], RISK_EXCEEDED)
        self.assertTrue(all(r.move_id == "M1" for r in rejections))


if __name__ == "__main__":
    unittest.main()
