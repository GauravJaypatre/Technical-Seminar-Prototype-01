"""Stage 1: Repository Profiler & Requirement Contract Agent.

Extracts a formal RequirementContract from the raw issue and repository profile,
producing the initial bounded EvidenceBaton B0 (budget <= 500 tokens).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.agents.base import BaseAgent
from src.agents.prompts import PROFILER_SYSTEM_PROMPT
from src.interfaces import IRepositoryProfiler, RepoProfile
from src.model.inference import InferenceEngine
from src.schemas import EvidenceBaton, RequirementContract, estimate_token_count

logger = logging.getLogger(__name__)


class ProfilerAgent(BaseAgent[RequirementContract]):
    """Stage 1b Agent: Converts issue description and profile into a formal RequirementContract."""

    def __init__(
        self,
        engine: InferenceEngine,
        profiler: IRepositoryProfiler | None = None,
    ) -> None:
        super().__init__(
            engine=engine,
            system_prompt=PROFILER_SYSTEM_PROMPT,
            stage_name="stage1_profiler",
        )
        self.profiler = profiler

    def deterministic_fallback(self, issue: str, profile: RepoProfile | None = None) -> RequirementContract:
        """Deterministic fallback when SLM inference fails or produces invalid JSON."""
        logger.warning("Using deterministic fallback to construct RequirementContract.")
        lines = [line.strip() for line in issue.splitlines() if line.strip()]
        first_line = lines[0] if lines else "Unspecified defect"
        lang = profile.language if profile else "python"

        # Determine task type heuristically
        lower_issue = issue.lower()
        if "feature" in lower_issue or "add support" in lower_issue:
            task_type = "feature"
        elif "refactor" in lower_issue or "cleanup" in lower_issue:
            task_type = "refactor"
        else:
            task_type = "bug_fix"

        return RequirementContract(
            task_type=task_type,
            observed_behavior=f"Reported issue: {first_line[:200]}",
            expected_behavior=f"Resolve defect so that: {first_line[:200]} is correctly handled",
            inviolable_constraints=[
                f"Maintain compatibility with {lang} runtime",
                "Do not introduce regressions in existing test suite",
            ],
            acceptance_oracles=[f"test_{task_type}_verification"],
        )

    def extract_contract(
        self,
        issue: str,
        profile: RepoProfile,
    ) -> RequirementContract:
        """Call SLM to extract RequirementContract, falling back if parsing fails."""
        user_prompt = (
            f"Repository Language: {profile.language}\n"
            f"Build System: {profile.build_system}\n"
            f"Test Framework: {profile.test_framework}\n\n"
            f"Issue Description:\n{issue.strip()}\n"
        )

        try:
            contract = self.engine.generate_json(
                system=self.system_prompt,
                user=user_prompt,
                schema=RequirementContract,
                max_tokens=self.engine.config.max_tokens_profiler,
                stage_name=self.stage_name,
            )
            if isinstance(contract, RequirementContract):
                return contract
            return RequirementContract.model_validate(contract)
        except Exception as e:
            logger.error("ProfilerAgent SLM generation failed: %s. Engaging fallback.", e)
            return self.deterministic_fallback(issue, profile)

    def create_initial_baton(self, contract: RequirementContract) -> EvidenceBaton:
        """Creates initial Baton B0 and verifies <= 500 token limit."""
        b0 = EvidenceBaton(
            requirement=contract,
            candidate_files=[],
            suspicious_ranges=[],
            hypotheses=[],
            verified_hypothesis=None,
            retry_traces=[],
            confidence_scores={},
        )
        projected_b0 = b0.project(max_tokens=self.engine.config.initial_baton_max_tokens)
        token_count = projected_b0.token_count()
        logger.info("Stage 1 produced initial Baton B0 (%d tokens <= 500 cap)", token_count)
        return projected_b0

    def run(self, baton: EvidenceBaton, context: str, **kwargs: Any) -> RequirementContract:
        """Run agent conforming to BaseAgent interface."""
        issue = kwargs.get("issue", "")
        profile = kwargs.get("profile") or RepoProfile(repo_path=".")
        return self.extract_contract(issue, profile)
