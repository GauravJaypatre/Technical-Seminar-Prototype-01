"""Stage 4: Hypothesis Agent.

Formulates at most 3 mutually exclusive, falsifiable root-cause hypotheses,
each with a concrete failure mechanism and empirical falsification test condition.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.agents.prompts import HYPOTHESIS_SYSTEM_PROMPT
from src.model.inference import InferenceEngine
from src.schemas import EvidenceBaton, Hypothesis

logger = logging.getLogger(__name__)


class HypothesisBatch(BaseModel):
    """Container for batch hypothesis generation."""
    hypotheses: list[Hypothesis] = Field(default_factory=list)


class HypothesisAgent(BaseAgent[list[Hypothesis]]):
    """Stage 4 Agent: Generates <= 3 falsifiable hypotheses."""

    def __init__(self, engine: InferenceEngine) -> None:
        super().__init__(
            engine=engine,
            system_prompt=HYPOTHESIS_SYSTEM_PROMPT,
            stage_name="stage4_hypothesis",
        )

    def _repair_and_validate(self, batch: HypothesisBatch) -> list[Hypothesis]:
        """Ensures <= 3 hypotheses, valid failure mechanisms, and falsification tests."""
        valid: list[Hypothesis] = []

        for idx, h in enumerate(batch.hypotheses[:3]):
            hid = h.id or f"H{idx + 1}"
            statement = h.statement.strip() or f"Defect in candidate files for H{idx + 1}"
            failure_mechanism = (
                h.failure_mechanism.strip()
                or "Inconsistent state or unexpected exception encountered during execution"
            )
            falsification_test = (
                h.falsification_test.strip()
                or "Falsified if passing correct inputs fails or test oracle fails to reproduce"
            )
            score = max(0.0, min(1.0, float(h.score if h.score > 0 else 0.8 - idx * 0.15)))

            valid.append(
                Hypothesis(
                    id=hid,
                    statement=statement,
                    failure_mechanism=failure_mechanism,
                    falsification_test=falsification_test,
                    status=h.status or "PROPOSED",
                    score=round(score, 2),
                )
            )

        if not valid:
            logger.warning("Stage 4 produced 0 hypotheses. Synthesizing default hypothesis.")
            valid.append(
                Hypothesis(
                    id="H1",
                    statement="Observed behavior is caused by a logic or condition error in candidate files.",
                    failure_mechanism="Unsatisfied condition or incorrect calculation logic.",
                    falsification_test="Falsified if isolated reproduction script fails to trigger bug.",
                    status="PROPOSED",
                    score=0.75,
                )
            )

        return valid[:3]

    def run(self, baton: EvidenceBaton, context: str, **kwargs: Any) -> list[Hypothesis]:
        """Generate hypotheses given baton and compiled context."""
        user_prompt = (
            f"Evidence Baton (Contract & Localized Files):\n"
            f"{baton.to_json()}\n\n"
            f"Compiled Code Context:\n"
            f"{context}\n\n"
            f"Generate at most 3 mutually exclusive, falsifiable root-cause hypotheses."
        )

        try:
            batch = self.engine.generate_json(
                system=self.system_prompt,
                user=user_prompt,
                schema=HypothesisBatch,
                max_tokens=self.engine.config.max_tokens_hypothesis,
                stage_name=self.stage_name,
            )
            if not isinstance(batch, HypothesisBatch):
                batch = HypothesisBatch.model_validate(batch)
        except Exception as e:
            logger.error("Stage 4 SLM generation failed: %s. Using default hypothesis.", e)
            batch = HypothesisBatch(hypotheses=[])

        hypotheses = self._repair_and_validate(batch)
        logger.info("Stage 4 generated %d hypotheses", len(hypotheses))
        return hypotheses
