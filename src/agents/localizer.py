"""Stage 3: Fault Localizer Agent.

Pinpoints candidate files (k<=5) and suspicious line ranges [start, end]
with confidence values, validating every entry against the real repository
to eliminate hallucinated files and line ranges.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.agents.base import BaseAgent
from src.agents.prompts import LOCALIZER_SYSTEM_PROMPT
from src.interfaces import ITestLogCorrelator
from src.model.inference import InferenceEngine
from src.schemas import EvidenceBaton, LocalizationResult, SuspiciousRange

logger = logging.getLogger(__name__)


class LocalizerAgent(BaseAgent[LocalizationResult]):
    """Stage 3 Agent: Fault Localization with hallucination filtering."""

    def __init__(
        self,
        engine: InferenceEngine,
        correlator: ITestLogCorrelator | None = None,
    ) -> None:
        super().__init__(
            engine=engine,
            system_prompt=LOCALIZER_SYSTEM_PROMPT,
            stage_name="stage3_localizer",
        )
        self.correlator = correlator

    def _validate_and_filter(
        self,
        raw_result: LocalizationResult,
        repo_path: Path,
    ) -> LocalizationResult:
        """Discards files and ranges that do not exist in the target repository."""
        valid_files: list[str] = []
        valid_ranges: list[SuspiciousRange] = []

        for f in raw_result.candidate_files:
            clean_f = f.replace("\\", "/").strip().lstrip("./")
            target = repo_path / clean_f
            if target.exists() and target.is_file():
                if clean_f not in valid_files:
                    valid_files.append(clean_f)
            else:
                logger.warning("Filtered hallucinated file from localizer: %s", f)

        for r in raw_result.suspicious_ranges:
            clean_f = r.file.replace("\\", "/").strip().lstrip("./")
            target = repo_path / clean_f
            if target.exists() and target.is_file():
                try:
                    num_lines = sum(1 for _ in open(target, "r", encoding="utf-8", errors="replace"))
                except Exception:
                    num_lines = 1000

                # Sanitize and clamp bounds
                start = max(1, min(r.start, num_lines))
                end = max(start, min(r.end, num_lines))

                valid_ranges.append(
                    SuspiciousRange(
                        file=clean_f,
                        start=start,
                        end=end,
                        confidence=r.confidence,
                    )
                )
                if clean_f not in valid_files:
                    valid_files.append(clean_f)
            else:
                logger.warning("Filtered hallucinated range in non-existent file: %s", r.file)

        return LocalizationResult(
            candidate_files=valid_files[:5],
            suspicious_ranges=valid_ranges[:5],
        )

    def run(
        self,
        baton: EvidenceBaton,
        context: str,
        repo_path: Path | str = ".",
        **kwargs: Any,
    ) -> LocalizationResult:
        """Execute Stage 3 Fault Localization."""
        resolved_repo = Path(repo_path).resolve()

        user_prompt = (
            f"Evidence Baton (Contract & Task):\n"
            f"{baton.to_json()}\n\n"
            f"Compiled Code Context:\n"
            f"{context}\n\n"
            f"Identify suspicious candidate files (max 5) and line ranges."
        )

        try:
            raw_result = self.engine.generate_json(
                system=self.system_prompt,
                user=user_prompt,
                schema=LocalizationResult,
                max_tokens=self.engine.config.max_tokens_localizer,
                stage_name=self.stage_name,
            )
            if not isinstance(raw_result, LocalizationResult):
                raw_result = LocalizationResult.model_validate(raw_result)
        except Exception as e:
            logger.error("Stage 3 SLM generation failed: %s. Using empty localization result.", e)
            raw_result = LocalizationResult(candidate_files=[], suspicious_ranges=[])

        # Validate against repository to eliminate hallucinations
        filtered = self._validate_and_filter(raw_result, resolved_repo)

        # Disambiguation / fallback check
        needs_disambiguation = (
            not filtered.candidate_files
            or (filtered.suspicious_ranges and max(r.confidence for r in filtered.suspicious_ranges) < 0.6)
        )

        if needs_disambiguation and self.correlator and baton.retry_traces:
            trace = baton.retry_traces[0]
            logger.info("Localizer confidence ambiguous. Invoking TestLogCorrelator.")
            correlated_ranges = self.correlator.correlate(trace, resolved_repo)
            if correlated_ranges:
                for cr in correlated_ranges:
                    if cr.file not in filtered.candidate_files:
                        filtered.candidate_files.append(cr.file)
                    filtered.suspicious_ranges.append(cr)
                filtered = self._validate_and_filter(filtered, resolved_repo)

        logger.info(
            "Stage 3 localized %d candidate files and %d ranges",
            len(filtered.candidate_files),
            len(filtered.suspicious_ranges),
        )
        return filtered
