"""Freight disruption recovery plan orchestration.

Public surface:

* :class:`RecoveryOrchestrator` — decide safely, publish atomically;
* :class:`ResourceStore` / :class:`InMemoryAuditLedger` — transactional target
  and the append-only reason ledger;
* :class:`RecoveryRequest` / :class:`DisruptedTrain` / :class:`Network` and the
  rest of the immutable domain model in :mod:`domain`;
* :class:`RecoveryOutcome` / :class:`DecisionCode` — terminal decisions.
"""
from __future__ import annotations

from .atomic import (
    PublicationFailed,
    ResourceContention,
    ResourceStore,
    RollbackReport,
    VersionConflict,
)
from .domain import (
    AuditCode,
    Blockage,
    Crew,
    CustomerCommitment,
    DecisionCode,
    DisruptedTrain,
    DutyLimits,
    Evaluation,
    Finding,
    Locomotive,
    Network,
    OptionKind,
    Railcar,
    RecoveryOption,
    RecoveryPlan,
    RecoveryRequest,
    Segment,
    Severity,
)
from .orchestrator import (
    AuditEntry,
    InMemoryAuditLedger,
    RecoveryOrchestrator,
    RecoveryOutcome,
)

__all__ = [
    "AuditCode", "AuditEntry", "Blockage", "Crew", "CustomerCommitment",
    "DecisionCode", "DisruptedTrain", "DutyLimits", "Evaluation", "Finding",
    "InMemoryAuditLedger", "Locomotive", "Network", "OptionKind",
    "PublicationFailed", "Railcar", "RecoveryOption", "RecoveryOrchestrator",
    "RecoveryOutcome", "RecoveryPlan", "RecoveryRequest",
    "ResourceContention", "ResourceStore", "RollbackReport", "Segment",
    "Severity", "VersionConflict",
]
