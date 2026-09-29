"""Stub implementation of Stage 5 Evidence Gate and Falsification Critic.

# TODO(member-3): Implement deterministic AST/type-check verification and conditional falsification critic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.interfaces import CriticVerdict, GateStatus, IEvidenceGate, IFalsificationCritic
from src.schemas import EvidenceBaton, Hypothesis

logger = logging.getLogger(__name__)


class FalsificationCriticStub(IFalsificationCritic):
    """Stub implementation of FalsificationCritic."""

    def critique(self, hypothesis: Hypothesis, evidence: str) -> CriticVerdict:
        # TODO(member-3): Implement SLM-based conditional falsification evaluation
        return CriticVerdict(
            verdict="SUPPORTED",
            reasoning=f"Hypothesis {hypothesis.id} aligns with available evidence.",
            falsification_evidence="No falsifying counterexamples detected in stub evaluation.",
        )


class EvidenceGateStub(IEvidenceGate):
    """Stub implementation of Stage 5 Evidence Gate."""

    def __init__(self, critic: IFalsificationCritic | None = None) -> None:
        self.critic = critic or FalsificationCriticStub()

    def evaluate(
        self,
        baton: EvidenceBaton,
        repo: str | Path
    ) -> tuple[Hypothesis, GateStatus]:
        # TODO(member-3): Implement deterministic static verification + critic routing on CONFLICT
        if not baton.hypotheses:
            # Fallback hypothesis if none generated
            fallback = Hypothesis(
                id="H0_fallback",
                statement="Direct patch required based on requirement contract.",
                failure_mechanism="Unspecified defect in candidate files.",
                falsification_test="Run test suite acceptance oracles.",
                status="SUPPORTED",
                score=0.5,
            )
            return fallback, GateStatus.SUPPORTED

        # If hypotheses have scores, pick the highest
        ranked = sorted(baton.hypotheses, key=lambda h: h.score, reverse=True)
        winner = ranked[0]
        winner.status = "SUPPORTED"

        status = GateStatus.SUPPORTED
        if len(baton.hypotheses) > 1 and abs(baton.hypotheses[0].score - baton.hypotheses[1].score) < 0.05:
            # Ambiguity triggers conflict resolution
            verdict = self.critic.critique(winner, str(baton.candidate_files))
            if verdict.verdict == "SUPPORTED":
                status = GateStatus.RESOLVED_CONFLICT
            else:
                status = GateStatus.CONFLICT

        logger.info("EvidenceGateStub selected winning hypothesis: %s (status: %s)", winner.id, status)
        return winner, status
