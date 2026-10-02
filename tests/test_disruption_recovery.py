import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from railforge.api import RailForgeService
from railforge.disruption_recovery import candidates, evaluator, risk, safety, scheduler
from railforge.disruption_recovery.atomic import ResourceStore
from railforge.disruption_recovery.domain import (
    AuditCode,
    Blockage,
    Crew,
    CustomerCommitment,
    DecisionCode,
    DisruptedTrain,
    Locomotive,
    OptionKind,
    Railcar,
    RecoveryRequest,
    Severity,
)
from railforge.disruption_recovery.orchestrator import RecoveryOrchestrator
from railforge.disruption_recovery.sample_world import T0, build_request

U = timezone.utc


def codes(findings):
    return {f.code for f in findings}


class CandidateAndGateTests(unittest.TestCase):
    def test_wait_and_detours_are_generated_but_shortcut_is_safe_to_consider(self):
        req = build_request("standard")
        options = candidates.generate(req)
        kinds = {(o.kind, o.path) for o in options}
        self.assertIn((OptionKind.WAIT, ("A", "B", "E")), kinds)
        self.assertIn((OptionKind.REROUTE, ("A", "D", "E")), kinds)
        self.assertIn((OptionKind.REROUTE, ("A", "C", "E")), kinds)
        # No reroute traverses the still-blocked segment.
        for o in options:
            if o.kind is OptionKind.REROUTE:
                self.assertNotIn("BE", {s.segment_id for s in req.network.path_segments(o.path)})

    def test_wait_never_enters_blocked_segment_before_clearance(self):
        req = build_request("standard")
        wait = next(o for o in candidates.generate(req) if o.kind is OptionKind.WAIT)
        itin = scheduler.schedule(req, wait)
        blockage = req.network.blockage_on("BE")
        be = next(m for m in itin.movements if m.segment.segment_id == "BE")
        self.assertTrue(be.waiting_on_blockage)
        self.assertGreaterEqual(be.enter_at, blockage.estimated_clear)

    def test_hard_gate_blocks_certification_traction_hazmat_and_hours(self):
        req = build_request("standard")
        by_id = {}
        for o in candidates.generate(req):
            ev = evaluator.evaluate_option(req, o)
            by_id.setdefault(ev.option.option_id, ev)

        # Original units on the detour: engineer not certified for AD/DE.
        ev = next(e for e in by_id.values()
                  if e.option.power.head == ("L1", "L1B") and e.option.path == ("A", "D", "E")
                  and not e.option.crew_changes)
        self.assertIn("SAFETY_LOCO_CERTIFICATION", codes(ev.hard_rejections))
        self.assertFalse(ev.accepted)

        # L2 without the D helper: grade exceeds powered effort incl. margin.
        ev = next(e for e in by_id.values()
                  if e.option.power.head == ("L2",) and not e.option.power.helpers
                  and e.option.path == ("A", "D", "E") and not e.option.crew_changes)
        self.assertIn("SAFETY_TRACTION", codes(ev.hard_rejections))

        # Hazmat placards through the AC/CE prohibition are refused.
        ev = next(e for e in by_id.values()
                  if e.option.path == ("A", "C", "E") and e.option.power.head == ("L4",))
        self.assertIn("SAFETY_HAZMAT_PROHIBITED", codes(ev.hard_rejections))

        # L2 + helper but no handoff: crew K1 is over hours of service.
        ev = next(e for e in by_id.values()
                  if e.option.power.head == ("L2",) and e.option.power.helpers == ("L3",)
                  and not e.option.crew_changes)
        self.assertIn("SAFETY_HOURS_OF_SERVICE", codes(ev.hard_rejections))

    def test_safety_cannot_be_traded_for_a_local_on_time_arrival(self):
        """Light hazmat train: the only on-time path is the prohibited shortcut.

        The detour misses the deadline; the wait is worse.  The shortcut would
        arrive 2.5h early but is hazmat-prohibited, so the system must refuse
        every alternative rather than waive the safety commitment.
        """
        req = build_request("standard")
        light_cars = tuple(
            Railcar(f"C{i:02d}", 1000.0, hazmat=True,
                    brake_valid_until=T0 + timedelta(hours=30)) for i in range(1, 4))
        tight = tuple(CustomerCommitment(c.car_id, "E", T0 + timedelta(hours=4),
                                         max_tolerated_lateness_minutes=30)
                      for c in light_cars)
        train = replace(req.train, cars=light_cars, commitments=tight)
        req_tight = replace(req, train=train)

        shortcut_evals = [e for e in (evaluator.evaluate_option(req_tight, o)
                                      for o in candidates.generate(req_tight))
                          if e.option.path == ("A", "C", "E")]
        self.assertTrue(shortcut_evals)
        for ev in shortcut_evals:
            self.assertIn("SAFETY_HAZMAT_PROHIBITED", codes(ev.hard_rejections))

        outcome = RecoveryOrchestrator().run(req_tight)
        self.assertIs(outcome.decision, DecisionCode.NO_CONTROLLED_ALTERNATIVE)
        self.assertIsNone(outcome.plan)
        rejected_paths = {tuple(r["path"]) for r in _abort_reasons(outcome)}
        self.assertIn(("A", "C", "E"), rejected_paths)

    def test_same_light_train_without_hazmat_takes_the_shortcut(self):
        """Positive control: when the prohibition does not apply, the shortcut
        is a legitimate controlled alternative."""
        req = build_request("standard")
        light_cars = tuple(
            Railcar(f"C{i:02d}", 1000.0, brake_valid_until=T0 + timedelta(hours=30))
            for i in range(1, 4))
        tight = tuple(CustomerCommitment(c.car_id, "E", T0 + timedelta(hours=4),
                                         max_tolerated_lateness_minutes=30)
                      for c in light_cars)
        train = replace(req.train, cars=light_cars, commitments=tight)
        crew = replace(train.crew, qualified_hazmat=True)
        train = replace(train, crew=crew)
        outcome = RecoveryOrchestrator().run(replace(req, train=train))
        self.assertIs(outcome.decision, DecisionCode.PUBLISHED)
        self.assertEqual(outcome.plan.path, ("A", "C", "E"))
        self.assertEqual(outcome.plan.max_lateness_minutes, 0)

    def test_wait_is_refused_by_risk_gate_not_safety(self):
        req = build_request("standard")
        wait = next(o for o in candidates.generate(req) if o.kind is OptionKind.WAIT)
        ev = evaluator.evaluate_option(req, wait)
        self.assertFalse(ev.hard_rejections)          # waiting is physically legal
        self.assertTrue(ev.risk_findings)             # but uncontrolled
        self.assertEqual(codes(ev.risk_findings), {"RISK_TOLERANCE_EXCEEDED"})
        breakdown = risk.assess(req, wait, scheduler.schedule(req, wait))
        self.assertGreater(breakdown.blockage, 0.99)  # 8h clearance uncertainty
        self.assertGreater(breakdown.crew, 0.9)       # wafer-thin duty reserve

    def test_expired_brake_inspection_rejects_all_routings(self):
        req = build_request("standard", brake_hours=4)  # expires 12:00
        outcome = RecoveryOrchestrator().run(req)
        self.assertIs(outcome.decision, DecisionCode.NO_CONTROLLED_ALTERNATIVE)
        all_codes = {f.code for ev in outcome.evaluations for f in ev.hard_rejections}
        self.assertIn("SAFETY_BRAKE_INSPECTION", all_codes)

    def test_safety_gate_is_inclusive_at_the_boundary(self):
        """Inspection valid exactly at arrival passes (< is the rejection)."""
        req = build_request("standard", brake_hours=5)  # 13:00 == arrival
        option = next(o for o in candidates.generate(req)
                      if o.power.head == ("L2",) and o.power.helpers == ("L3",)
                      and o.crew_changes == ("D",))
        itin = scheduler.schedule(req, option)
        hard = safety.evaluate(req, option, itin)
        self.assertEqual(hard, [])

    def test_tightening_risk_appetite_closes_the_detour(self):
        # The controlled detour carries ~0.084 traction-margin risk.
        req = replace(build_request("standard"), max_acceptable_risk=0.05)
        outcome = RecoveryOrchestrator().run(req)
        self.assertIs(outcome.decision, DecisionCode.NO_CONTROLLED_ALTERNATIVE)
        detour = [r for r in _abort_reasons(outcome)
                  if tuple(r["path"]) == ("A", "D", "E")
                  and r["option_id"].endswith(("R17",))]
        self.assertTrue(any(r["rejected_by"] == "risk" for r in detour))


class SelectionTests(unittest.TestCase):
    def test_only_one_controlled_alternative_and_it_meets_deadline(self):
        req = build_request("standard")
        outcome = RecoveryOrchestrator().run(req)
        self.assertIs(outcome.decision, DecisionCode.PUBLISHED)
        plan = outcome.plan
        self.assertEqual(plan.path, ("A", "D", "E"))
        self.assertEqual(plan.power_ids, ("L2", "L3"))
        self.assertEqual(plan.crew_changes, ("D",))
        self.assertEqual(plan.max_lateness_minutes, 0)
        self.assertEqual(plan.arrive_at, T0 + timedelta(hours=5))  # 13:00
        self.assertLessEqual(plan.risk_score, req.max_acceptable_risk)
        accepted = [e for e in outcome.evaluations if e.accepted]
        self.assertEqual(len(accepted), 1)

    def test_selection_orders_by_broken_commitments_then_lateness_then_risk(self):
        req = build_request("standard")
        options = candidates.generate(req)
        evals = [evaluator.evaluate_option(req, o) for o in options]
        # Determinism: select() is stable and never picks a rejected option.
        first = evaluator.select(evals)
        self.assertIsNotNone(first)
        self.assertTrue(first.accepted)
        for _ in range(3):
            self.assertIs(evaluator.select(evals), first)


class AtomicPublicationTests(unittest.TestCase):
    def setUp(self):
        self.req = build_request("standard")

    def test_plan_and_all_reservations_publish_as_one_unit(self):
        orch = RecoveryOrchestrator()
        outcome = orch.run(self.req)
        snap = orch.store.snapshot()
        plan_id = outcome.plan.plan_id
        owners = {k: v["reserved_by"] for k, v in snap["resources"].items()}
        self.assertEqual(owners, {
            "train:7MS1-standard": plan_id,
            "locomotive:L2": plan_id,
            "locomotive:L3": plan_id,
            "route_segment:AD": plan_id,
            "route_segment:DE": plan_id,
        })
        for v in snap["resources"].values():
            self.assertEqual(v["version"], 1)
        self.assertIn(plan_id, snap["plans"])
        codes_ = orch.ledger.codes()
        self.assertEqual(codes_[0], AuditCode.REQUEST_RECEIVED)
        self.assertIn(AuditCode.OPTION_SELECTED, codes_)
        self.assertEqual(codes_[-1], AuditCode.PLAN_COMMITTED)

    def test_publication_hook_failure_rolls_back_byte_for_byte(self):
        def hook(publication, resources):
            raise RuntimeError("signaling outbox unavailable")

        orch = RecoveryOrchestrator(publication_hook=hook)
        before = orch.store.snapshot()
        outcome = orch.run(self.req)
        after = orch.store.snapshot()

        self.assertIs(outcome.decision, DecisionCode.PUBLICATION_FAILED)
        self.assertIsNone(outcome.plan)
        self.assertEqual(after, before)  # no partial reservation, no plan row
        self.assertIn("signaling outbox unavailable", outcome.rollback_reason)
        tail = orch.ledger.codes()[-2:]
        self.assertEqual(tail, [AuditCode.PLAN_PUBLISH_FAILED, AuditCode.PLAN_ROLLED_BACK])
        failed = orch.ledger.entries[-2]
        self.assertTrue(failed.detail["touched"])  # what would have changed

    def test_resource_contention_rolls_back_without_touching_holder(self):
        store = ResourceStore()
        store.ensure("locomotive", "L2").reserved_by = "plan-OTHER"
        orch = RecoveryOrchestrator(store=store)
        before = store.snapshot()
        outcome = orch.run(self.req)

        self.assertIs(outcome.decision, DecisionCode.PUBLICATION_FAILED)
        self.assertIn("locomotive:L2 already reserved by plan-OTHER",
                      outcome.rollback_reason)
        self.assertEqual(store.snapshot(), before)
        self.assertIsNone(store.get("train", "7MS1-standard"))  # nothing leaked
        self.assertEqual(store.get("locomotive", "L2").reserved_by, "plan-OTHER")

    def test_concurrent_version_change_aborts_publication(self):
        store = ResourceStore()
        store.ensure("route_segment", "AD")
        original = ResourceStore.commit

        def racing(self_, shadow, plan_id, record, at, publication_hook=None):
            r = store.get("route_segment", "AD")
            from railforge.disruption_recovery.atomic import Resource as _Resource
            store.resources["route_segment:AD"] = _Resource(
                r.kind, r.resource_id, r.version + 1, "plan-RACE", {})
            return original(self_, shadow, plan_id, record, at, publication_hook)

        ResourceStore.commit = racing
        try:
            outcome = RecoveryOrchestrator(store=store).run(self.req)
        finally:
            ResourceStore.commit = original

        self.assertIs(outcome.decision, DecisionCode.PUBLICATION_CONFLICT)
        self.assertEqual(store.snapshot()["plans"], {})
        self.assertEqual(store.get("route_segment", "AD").reserved_by, "plan-RACE")

    def test_replay_with_same_request_id_is_idempotent(self):
        orch = RecoveryOrchestrator()
        first = orch.run(self.req)
        second = orch.run(self.req)
        self.assertIs(first, second)

    def test_no_controlled_alternative_publishes_nothing_but_keeps_reasons(self):
        orch = RecoveryOrchestrator()
        outcome = orch.run(build_request("heavy"))
        self.assertIs(outcome.decision, DecisionCode.NO_CONTROLLED_ALTERNATIVE)
        self.assertEqual(orch.store.snapshot()["plans"], {})
        self.assertTrue(all(v["reserved_by"] is None
                            for v in orch.store.snapshot()["resources"].values()))
        abort = orch.ledger.entries[-1]
        self.assertEqual(abort.code, AuditCode.RECOVERY_ABORTED)
        self.assertGreater(len(abort.detail["reasons"]), 0)
        for reason in abort.detail["reasons"]:
            self.assertIn(reason["rejected_by"], {"safety", "risk", "commitment"})
            self.assertTrue(reason["codes"])

    def test_severe_weather_can_close_the_last_controlled_option(self):
        req = replace(build_request("standard"),
                      weather_severity={"AD": 1.0, "DE": 1.0},
                      max_acceptable_risk=0.30)
        outcome = RecoveryOrchestrator().run(req)
        self.assertIs(outcome.decision, DecisionCode.NO_CONTROLLED_ALTERNATIVE)
        # The detour died on risk (weather), not on a safety waiver.
        detour_reasons = [r for r in _abort_reasons(outcome) if tuple(r["path"]) == ("A", "D", "E")]
        self.assertTrue(any("RISK_TOLERANCE_EXCEEDED" in r["codes"] for r in detour_reasons))


class ServiceIntegrationTests(unittest.TestCase):
    def test_service_projects_published_plan_onto_control_plane(self):
        svc = RailForgeService()
        receipt = svc.recover_disruption(build_request("standard"))
        self.assertEqual(receipt.event.topic, "disruption.recovery.plan")
        self.assertEqual(receipt.event.payload["status"], "committed")
        self.assertEqual(receipt.event.payload["control_path"], "atomic")
        self.assertEqual(receipt.result["decision"], "published")
        self.assertTrue(svc.workflow().stream.verify_chain())
        self.assertTrue(svc.workflow().trail.verify())
        # Control-plane projection is idempotent for the same request.
        again = svc.recover_disruption(build_request("standard"))
        self.assertIs(again, receipt)

    def test_service_projection_records_rollback_reason(self):
        svc = RailForgeService()

        def hook(publication, resources):
            raise RuntimeError("outbox down")

        svc._recovery = RecoveryOrchestrator(publication_hook=hook)
        receipt = svc.recover_disruption(build_request("standard"))
        self.assertEqual(receipt.result["decision"], "publication_failed")
        self.assertIn("outbox down", receipt.result["rollback_reason"])
        # Recovery-side refusal still leaves the platform chains intact.
        self.assertTrue(svc.workflow().stream.verify_chain())
        self.assertTrue(svc.workflow().trail.verify())


def _abort_reasons(outcome):
    entry = next(e for e in outcome.audit if e.code is AuditCode.RECOVERY_ABORTED)
    return entry.detail["reasons"]


if __name__ == "__main__":
    unittest.main()
