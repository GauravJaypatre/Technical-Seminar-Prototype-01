"""Stage 6: Patch Generator Agent.

Produces a minimal unified diff based on the verified root-cause hypothesis,
localized code context, and RequirementContract constraints, with support
for test-driven retry loops incorporating failing tracebacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

from src.agents.base import BaseAgent
from src.agents.prompts import PATCH_SYSTEM_PROMPT
from src.interfaces import IRepoSandbox
from src.model.inference import InferenceEngine
from src.schemas import EvidenceBaton

logger = logging.getLogger(__name__)


def extract_unified_diff(raw_text: str) -> str:
    """Extracts unified diff content from raw model generation, stripping markdown fences."""
    text = raw_text.strip()

    # Extract from markdown block if present
    if "```" in text:
        match = re.search(r"```(?:diff|patch)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
        else:
            text = re.sub(r"^```(?:diff|patch)?", "", text)
            text = re.sub(r"```$", "", text).strip()

    # Look for start of diff
    diff_start = text.find("--- ")
    if diff_start == -1:
        diff_start = text.find("diff --git ")

    if diff_start != -1:
        text = text[diff_start:]

    return text.strip()


def is_valid_diff_format(diff_text: str) -> bool:
    """Validates that a string has standard unified diff structure."""
    if not diff_text:
        return False

    has_from = "--- " in diff_text or "diff --git " in diff_text
    has_to = "+++ " in diff_text
    has_hunk = "@@ " in diff_text

    return has_from and has_to and has_hunk


class PatchGeneratorAgent(BaseAgent[str]):
    """Stage 6 Agent: Synthesizes minimal unified diffs."""

    def __init__(
        self,
        engine: InferenceEngine,
        sandbox: IRepoSandbox | None = None,
    ) -> None:
        super().__init__(
            engine=engine,
            system_prompt=PATCH_SYSTEM_PROMPT,
            stage_name="stage6_patch_generator",
        )
        self.sandbox = sandbox

    def run(
        self,
        baton: EvidenceBaton,
        context: str,
        attempt: int = 1,
        **kwargs: Any,
    ) -> str:
        """Generate unified diff.
        
        Incorporates verified hypothesis, inviolable constraints, and any
        retry error traces from the baton.
        """
        req = baton.requirement
        hyp = baton.verified_hypothesis or (baton.hypotheses[0] if baton.hypotheses else None)

        constraints_str = "\n".join(f"- {c}" for c in req.inviolable_constraints) or "- None specified"
        hypothesis_str = (
            f"ID: {hyp.id}\nStatement: {hyp.statement}\nMechanism: {hyp.failure_mechanism}"
            if hyp
            else "Direct fix based on requirement contract"
        )

        retry_section = ""
        if baton.retry_traces:
            last_trace = baton.retry_traces[-1]
            retry_section = (
                f"\nCRITICAL: PREVIOUS ATTEMPT FAILED WITH ERROR:\n"
                f"{last_trace}\n"
                f"You MUST modify your solution to resolve this traceback.\n"
            )

        user_prompt = (
            f"=== REQUIREMENT CONTRACT ===\n"
            f"Task Type: {req.task_type}\n"
            f"Observed: {req.observed_behavior}\n"
            f"Expected: {req.expected_behavior}\n"
            f"Inviolable Constraints:\n{constraints_str}\n\n"
            f"=== VERIFIED HYPOTHESIS ===\n{hypothesis_str}\n\n"
            f"=== LOCALIZED FILES & CONTEXT ===\n{context}\n"
            f"{retry_section}\n"
            f"Generate MINIMAL unified diff fixing the issue. "
            f"Remember: If case normalization is needed, apply it to BOTH registration and lookup methods. "
            f"Output unified diff ONLY."
        )

        raw_output = self.engine.generate(
            system=self.system_prompt,
            user=user_prompt,
            max_tokens=self.engine.config.max_tokens_patch,
            stage_name=f"{self.stage_name}_attempt_{attempt}",
        )

        diff = extract_unified_diff(raw_output)

        if not is_valid_diff_format(diff):
            logger.warning("Generated patch output does not have standard diff headers. Attempting raw output.")
            # If line starts with --- and +++, use it anyway
            if "---" in diff and "+++" in diff:
                return diff

        logger.info("Stage 6 generated diff (%d chars, attempt %d)", len(diff), attempt)
        return diff
