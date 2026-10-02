"""End-to-end walkthrough of freight disruption recovery orchestration.

Run with:  PYTHONPATH=src python3 examples/disruption_recovery_demo.py

Three scenarios:

1. a controlled detour exists — one alternative is safe, risk-controlled and
   on time; the recovery plan is published atomically with power, route and
   crew reservations;
2. the only on-time move would breach a committed hazmat prohibition (and the
   controlled alternatives break customer windows) — the system holds and
   escalates instead of trading safety for punctality;
3. the signaling outbox fails during publication — the whole transaction is
   rolled back, state is unchanged, and the reason is retained for audit.
"""
from __future__ import annotations

from dataclasses import replace

from railforge.disruption_recovery import (
    AuditCode,
    DecisionCode,
    Railcar,
    CustomerCommitment,
    RecoveryOrchestrator,
)
from railforge.disruption_recovery.sample_world import T0, build_request
from datetime import timedelta


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def summarize(orch, outcome) -> None:
    print(f"decision          : {outcome.decision.value}")
    print(f"alternatives seen : {len(outcome.evaluations)}")
    rejected_safety = sum(1 for e in outcome.evaluations if e.hard_rejections)
    rejected_risk = sum(1 for e in outcome.evaluations if e.risk_findings)
    rejected_commit = sum(1 for e in outcome.evaluations if e.commitment_findings)
    accepted = sum(1 for e in outcome.evaluations if e.accepted)
    print(f"  rejected safety : {rejected_safety}")
    print(f"  rejected risk   : {rejected_risk}")
    print(f"  rejected window : {rejected_commit}")
    print(f"  accepted        : {accepted}")
    if outcome.plan is not None:
        p = outcome.plan
        print(f"plan              : {p.path} power={list(p.power_ids)} "
              f"crew_change={list(p.crew_changes)}")
        print(f"depart / arrive   : {p.depart_at:%H:%M} -> {p.arrive_at:%H:%M}  "
              f"risk={p.risk_score:.3f} max_late={p.max_lateness_minutes}m")
        held = sorted(k for k, v in orch.store.snapshot()["resources"].items()
                      if v["reserved_by"] == p.plan_id)
        print(f"atomic reservations: {held}")
    if outcome.rollback_reason:
        print(f"rollback reason   : {outcome.rollback_reason}")
    tail_codes = [a.code.value for a in outcome.audit[-2:]]
    print(f"audit tail        : {tail_codes}")


def main() -> None:
    # 1) controlled detour ------------------------------------------------
    banner("1. Controlled recovery around B-E blockage")
    orch = RecoveryOrchestrator()
    summarize(orch, orch.run(build_request("standard")))

    # 2) safety cannot be traded for a local on-time arrival ---------------
    banner("2. Only on-time move breaches a hazmat prohibition -> hold")
    req = build_request("standard")
    light = tuple(Railcar(f"C{i:02d}", 1000.0, hazmat=True,
                          brake_valid_until=T0 + timedelta(hours=30))
                  for i in range(1, 4))
    tight = tuple(CustomerCommitment(c.car_id, "E", T0 + timedelta(hours=4),
                                     max_tolerated_lateness_minutes=30)
                  for c in light)
    unsafe_req = replace(req, train=replace(req.train, cars=light, commitments=tight))
    orch2 = RecoveryOrchestrator()
    outcome2 = orch2.run(unsafe_req)
    summarize(orch2, outcome2)
    abort = next(a for a in outcome2.audit if a.code is AuditCode.RECOVERY_ABORTED)
    shortcut = next(r for r in abort.detail["reasons"]
                    if tuple(r["path"]) == ("A", "C", "E")
                    and "SAFETY_HAZMAT_PROHIBITED" in r["codes"])
    print(f"shortcut refusal  : {shortcut['rejected_by']} {shortcut['codes']}")
    print("store plans       :", list(orch2.store.snapshot()["plans"]), "(nothing published)")

    # 3) publication failure -> complete rollback --------------------------
    banner("3. Signaling outbox fails at publication -> full rollback")

    def outbox(publication, resources):
        raise RuntimeError("signaling outbox unavailable")

    orch3 = RecoveryOrchestrator(publication_hook=outbox)
    before = orch3.store.snapshot()
    outcome3 = orch3.run(build_request("standard"))
    summarize(orch3, outcome3)
    print("ledger identical  :", orch3.store.snapshot() == before)


if __name__ == "__main__":
    main()
