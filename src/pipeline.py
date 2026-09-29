"""AERS-SLM Orchestration Pipeline.

Coordinates the 6 sequential stages:
1. Repository Profiler & Requirement Contract (B0 <= 500 tokens)
2. Hybrid Context Compiler (budget <= 4,000 tokens)
3. Fault Localizer (top k<=5 files & line ranges) -> B1
4. Hypothesis Agent (max 3 mutually exclusive, falsifiable hypotheses) -> B2
5. Evidence Gate (deterministic tools + falsification critic) -> B3
6. Patch Generator with test-driven retry loop (max 2 retries, rollback on failure)

Inter-stage communication occurs strictly via the bounded EvidenceBaton (<= 1,200 tokens).
Structured events are recorded as JSON Lines in logs/.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any

from src.agents.hypothesis import HypothesisAgent
from src.agents.localizer import LocalizerAgent
from src.agents.patch_generator import PatchGeneratorAgent
from src.agents.profiler import ProfilerAgent
from src.config import AERSConfig, get_config
from src.interfaces import GateStatus
from src.model.inference import FakeLLM, InferenceEngine, UsageTracker
from src.registry import ComponentRegistry
from src.schemas import (
    BATON_HARD_CAP_TOKENS,
    INITIAL_BATON_MAX_TOKENS,
    EvidenceBaton,
    FailureDiagnosis,
    PatchResult,
    RequirementContract,
)

logger = logging.getLogger(__name__)


class PipelineLogger:
    """Writes structured execution records to JSONL files for observability and Member 4."""

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"pipeline_{timestamp}.jsonl"

    def log_event(
        self,
        stage: str,
        status: str,
        duration_ms: float,
        baton_tokens: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "baton_tokens": baton_tokens,
            "details": details or {},
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.warning("Failed to write to JSONL log file: %s", e)


class AERSPipeline:
    """Sequential pipeline orchestrating Member 1 SLM intelligence stages."""

    def __init__(
        self,
        config: AERSConfig | None = None,
        engine: InferenceEngine | None = None,
        registry: ComponentRegistry | None = None,
        profiler_agent: ProfilerAgent | None = None,
        localizer_agent: LocalizerAgent | None = None,
        hypothesis_agent: HypothesisAgent | None = None,
        patch_agent: PatchGeneratorAgent | None = None,
    ) -> None:
        self.config = config or get_config()
        self.tracker = UsageTracker()

        # Resolve inference engine
        if engine is not None:
            self.engine = engine
        elif self.config.dry_run:
            self.engine = FakeLLM()
            self.engine.config = self.config
        else:
            try:
                self.engine = InferenceEngine(config=self.config, tracker=self.tracker)
            except Exception as e:
                logger.warning("Could not initialize full Llama engine: %s. Using FakeLLM fallback.", e)
                self.engine = FakeLLM()
                self.engine.config = self.config

        self.registry = registry or ComponentRegistry()
        self.structured_logger = PipelineLogger(log_dir=self.config.log_dir)

        # Inject or instantiate agents
        self.profiler_agent = profiler_agent or ProfilerAgent(
            engine=self.engine,
            profiler=self.registry.profiler,
        )
        self.localizer_agent = localizer_agent or LocalizerAgent(
            engine=self.engine,
            correlator=self.registry.correlator,
        )
        self.hypothesis_agent = hypothesis_agent or HypothesisAgent(
            engine=self.engine,
        )
        self.patch_agent = patch_agent or PatchGeneratorAgent(
            engine=self.engine,
            sandbox=self.registry.sandbox,
        )

    def _check_context_overflow(
        self, stage: str, prompt_text: str, max_tokens: int, baton: EvidenceBaton
    ) -> FailureDiagnosis | None:
        """Verifies prompt does not exceed n_ctx - max_tokens."""
        if hasattr(self.engine, "runtime") and self.engine.runtime and self.engine.runtime.is_loaded():
            prompt_tokens = self.engine.runtime.count_tokens(prompt_text)
        else:
            prompt_tokens = len(prompt_text) // 4
        allowed = self.config.n_ctx - max_tokens
        if prompt_tokens > allowed:
            msg = (
                f"Context overflow in {stage}: prompt tokens ({prompt_tokens}) "
                f"exceed allowed budget ({allowed} = n_ctx {self.config.n_ctx} - max_tokens {max_tokens})."
            )
            logger.error(msg)
            return FailureDiagnosis(stage=stage, reason=msg, baton=baton)
        return None

    def run(
        self,
        issue: str,
        repo_path: str | Path = ".",
    ) -> PatchResult | FailureDiagnosis:
        """Executes the 6-stage sequential bug fixing pipeline."""
        repo = Path(repo_path).resolve()
        logger.info("Starting AERS-SLM pipeline for repo: %s", repo)
        pipeline_start = time.perf_counter()

        # Start background resident memory (RSS) sampling
        if hasattr(self.engine, "tracker") and hasattr(self.engine.tracker, "start_memory_monitor"):
            self.engine.tracker.start_memory_monitor()

        # Dummy baton for fallback diagnostics
        fallback_contract = RequirementContract(
            task_type="bug_fix",
            observed_behavior=issue[:100],
            expected_behavior="Resolve reported issue",
            inviolable_constraints=[],
            acceptance_oracles=[],
        )
        b0 = EvidenceBaton(requirement=fallback_contract)

        # ---------------------------------------------------------------------
        # Stage 1: Repository Profiler & Requirement Contract
        # ---------------------------------------------------------------------
        stage_start = time.perf_counter()
        try:
            overflow1 = self._check_context_overflow("Stage 1: Profiler & Contract", issue, self.config.max_tokens_profiler, b0)
            if overflow1:
                return overflow1

            # Stage 1a: Deterministic Scan
            profile = self.registry.profiler.profile(repo)

            # Stage 1b: Requirement Contract Extraction
            contract = self.profiler_agent.extract_contract(issue, profile)

            # Create initial baton B0 (enforced <= 500 tokens)
            b0 = self.profiler_agent.create_initial_baton(contract)
            b0_tokens = b0.token_count()

            duration_ms = (time.perf_counter() - stage_start) * 1000.0
            meta1 = getattr(self.engine, "last_call_metadata", {})
            self.structured_logger.log_event(
                stage="Stage 1: Profiler & Contract",
                status="SUCCESS",
                duration_ms=duration_ms,
                baton_tokens=b0_tokens,
                details={
                    "task_type": contract.task_type,
                    "language": profile.language,
                    "raw_output": meta1.get("raw_output", ""),
                    "attempts": meta1.get("attempts", 1),
                    "first_try_valid": meta1.get("first_try_valid", True),
                    "prompt_tokens": meta1.get("prompt_tokens", 0),
                    "completion_tokens": meta1.get("completion_tokens", 0),
                    "peak_rss_mb": getattr(self.engine.tracker, "peak_rss_mb", 0.0),
                },
            )
        except Exception as e:
            logger.exception("Error in Stage 1")
            duration_ms = (time.perf_counter() - stage_start) * 1000.0
            self.structured_logger.log_event("Stage 1", "FAILED", duration_ms, details={"error": str(e)})
            return FailureDiagnosis(stage="Stage 1: Profiler & Contract", reason=str(e), baton=b0)

        # ---------------------------------------------------------------------
        # Stage 2: Hybrid Context Compiler (Member 2)
        # ---------------------------------------------------------------------
        stage_start = time.perf_counter()
        try:
            compiled_context = self.registry.context_compiler.compile(
                contract=b0.requirement,
                repo=repo,
                budget=self.config.context_budget,
            )
            duration_ms = (time.perf_counter() - stage_start) * 1000.0
            self.structured_logger.log_event(
                stage="Stage 2: Context Compiler",
                status="SUCCESS",
                duration_ms=duration_ms,
                baton_tokens=b0.token_count(),
                details={
                    "context_chars": len(compiled_context),
                    "peak_rss_mb": getattr(self.engine.tracker, "peak_rss_mb", 0.0),
                },
            )
        except Exception as e:
            logger.exception("Error in Stage 2")
            duration_ms = (time.perf_counter() - stage_start) * 1000.0
            self.structured_logger.log_event("Stage 2", "FAILED", duration_ms, details={"error": str(e)})
            return FailureDiagnosis(stage="Stage 2: Context Compiler", reason=str(e), baton=b0)

        # ---------------------------------------------------------------------
        # Stage 3: Fault Localizer
        # ---------------------------------------------------------------------
        stage_start = time.perf_counter()
        try:
            overflow3 = self._check_context_overflow("Stage 3: Fault Localizer", compiled_context, self.config.max_tokens_localizer, b0)
            if overflow3:
                return overflow3

            localization = self.localizer_agent.run(
                baton=b0,
                context=compiled_context,
                repo_path=repo,
            )
            b1 = b0.update(
                candidate_files=localization.candidate_files,
                suspicious_ranges=localization.suspicious_ranges,
            )
            b1_tokens = b1.token_count()

            duration_ms = (time.perf_counter() - stage_start) * 1000.0
            meta3 = getattr(self.engine, "last_call_metadata", {})
            self.structured_logger.log_event(
                stage="Stage 3: Fault Localizer",
                status="SUCCESS",
                duration_ms=duration_ms,
                baton_tokens=b1_tokens,
                details={
                    "candidate_files": b1.candidate_files,
                    "ranges_count": len(b1.suspicious_ranges),
                    "raw_output": meta3.get("raw_output", ""),
                    "attempts": meta3.get("attempts", 1),
                    "first_try_valid": meta3.get("first_try_valid", True),
                    "prompt_tokens": meta3.get("prompt_tokens", 0),
                    "completion_tokens": meta3.get("completion_tokens", 0),
                    "peak_rss_mb": getattr(self.engine.tracker, "peak_rss_mb", 0.0),
                },
            )
        except Exception as e:
            logger.exception("Error in Stage 3")
            duration_ms = (time.perf_counter() - stage_start) * 1000.0
            self.structured_logger.log_event("Stage 3", "FAILED", duration_ms, details={"error": str(e)})
            return FailureDiagnosis(stage="Stage 3: Fault Localizer", reason=str(e), baton=b0)

        # ---------------------------------------------------------------------
        # Stage 4: Hypothesis Agent
        # ---------------------------------------------------------------------
        stage_start = time.perf_counter()
        try:
            overflow4 = self._check_context_overflow("Stage 4: Hypothesis Agent", compiled_context, self.config.max_tokens_hypothesis, b1)
            if overflow4:
                return overflow4

            hypotheses = self.hypothesis_agent.run(baton=b1, context=compiled_context)
            b2 = b1.update(hypotheses=hypotheses)
            b2_tokens = b2.token_count()

            duration_ms = (time.perf_counter() - stage_start) * 1000.0
            meta4 = getattr(self.engine, "last_call_metadata", {})
            self.structured_logger.log_event(
                stage="Stage 4: Hypothesis Agent",
                status="SUCCESS",
                duration_ms=duration_ms,
                baton_tokens=b2_tokens,
                details={
                    "hypotheses_count": len(hypotheses),
                    "raw_output": meta4.get("raw_output", ""),
                    "attempts": meta4.get("attempts", 1),
                    "first_try_valid": meta4.get("first_try_valid", True),
                    "prompt_tokens": meta4.get("prompt_tokens", 0),
                    "completion_tokens": meta4.get("completion_tokens", 0),
                    "peak_rss_mb": getattr(self.engine.tracker, "peak_rss_mb", 0.0),
                },
            )
        except Exception as e:
            logger.exception("Error in Stage 4")
            duration_ms = (time.perf_counter() - stage_start) * 1000.0
            self.structured_logger.log_event("Stage 4", "FAILED", duration_ms, details={"error": str(e)})
            return FailureDiagnosis(stage="Stage 4: Hypothesis Agent", reason=str(e), baton=b1)

        # ---------------------------------------------------------------------
        # Stage 5: Evidence Gate (Member 3)
        # ---------------------------------------------------------------------
        stage_start = time.perf_counter()
        try:
            winner, gate_status = self.registry.evidence_gate.evaluate(baton=b2, repo=repo)
            b3 = b2.update(verified_hypothesis=winner)
            b3_tokens = b3.token_count()

            duration_ms = (time.perf_counter() - stage_start) * 1000.0
            self.structured_logger.log_event(
                stage="Stage 5: Evidence Gate",
                status="SUCCESS",
                duration_ms=duration_ms,
                baton_tokens=b3_tokens,
                details={"winner_id": winner.id, "gate_status": str(gate_status)},
            )
        except Exception as e:
            logger.exception("Error in Stage 5")
            duration_ms = (time.perf_counter() - stage_start) * 1000.0
            self.structured_logger.log_event("Stage 5", "FAILED", duration_ms, details={"error": str(e)})
            return FailureDiagnosis(stage="Stage 5: Evidence Gate", reason=str(e), baton=b2)

        # ---------------------------------------------------------------------
        # Stage 6: Patch Generator & Test Validation Retry Loop
        # ---------------------------------------------------------------------
        self.registry.sandbox.prepare(repo)
        sandbox_target = getattr(self.registry.sandbox, "sandbox_path", repo)
        max_retries = self.config.max_retries
        last_test_error = ""

        try:
            for attempt in range(1 + max_retries):
                attempt_num = attempt + 1
                stage_start = time.perf_counter()
                logger.info("Stage 6 Patch Generation attempt %d/%d", attempt_num, 1 + max_retries)

                try:
                    overflow6 = self._check_context_overflow("Stage 6: Patch Generator", compiled_context, self.config.max_tokens_patch, b3)
                    if overflow6:
                        return overflow6

                    # Generate minimal unified diff
                    patch = self.patch_agent.run(
                        baton=b3,
                        context=compiled_context,
                        attempt=attempt_num,
                    )

                    # Validate patch in sandbox / test runner
                    applied = self.registry.sandbox.apply_patch(patch)
                    test_result = self.registry.test_runner.run(sandbox_target, patch)

                    duration_ms = (time.perf_counter() - stage_start) * 1000.0

                    meta6 = getattr(self.engine, "last_call_metadata", {})
                    fuzzy_used = getattr(self.registry.sandbox, "fuzzy_applied", False)
                    canonical_diff = getattr(self.registry.sandbox, "canonical_diff", "") or patch

                    if test_result.passed:
                        self.structured_logger.log_event(
                            stage=f"Stage 6: Patch Attempt {attempt_num}",
                            status="PASSED",
                            duration_ms=duration_ms,
                            baton_tokens=b3.token_count(),
                            details={
                                "attempts": attempt_num,
                                "diff_size": len(canonical_diff),
                                "raw_output": meta6.get("raw_output", ""),
                                "fuzzy_applied": fuzzy_used,
                                "prompt_tokens": meta6.get("prompt_tokens", 0),
                                "completion_tokens": meta6.get("completion_tokens", 0),
                                "peak_rss_mb": getattr(self.engine.tracker, "peak_rss_mb", 0.0),
                            },
                        )
                        logger.info("Pipeline succeeded on attempt %d! (fuzzy_applied=%s)", attempt_num, fuzzy_used)
                        return PatchResult(
                            diff=canonical_diff,
                            passed=True,
                            attempts=attempt_num,
                            test_log=test_result.error_log or "All acceptance tests passed.",
                        )

                    # Test failed: record traceback, rollback repo sandbox
                    last_test_error = test_result.error_log
                    logger.warning(
                        "Test validation failed on attempt %d: %s. Rolling back.",
                        attempt_num,
                        last_test_error[:150],
                    )
                    self.registry.sandbox.rollback()

                    # Append error trace to baton (enforces hard cap <= 1,200 tokens)
                    b3 = b3.append_traceback(test_result.error_log)

                    self.structured_logger.log_event(
                        stage=f"Stage 6: Patch Attempt {attempt_num}",
                        status="RETRY_TRIGGERED",
                        duration_ms=duration_ms,
                        baton_tokens=b3.token_count(),
                        details={
                            "attempts": attempt_num,
                            "raw_output": meta6.get("raw_output", ""),
                            "fuzzy_applied": fuzzy_used,
                            "prompt_tokens": meta6.get("prompt_tokens", 0),
                            "completion_tokens": meta6.get("completion_tokens", 0),
                            "peak_rss_mb": getattr(self.engine.tracker, "peak_rss_mb", 0.0),
                            "error_snippet": last_test_error[:200],
                        },
                    )

                except Exception as e:
                    logger.exception("Exception during Stage 6 attempt %d", attempt_num)
                    self.registry.sandbox.rollback()
                    b3 = b3.append_traceback(str(e))

            # Rollback on final failure
            self.registry.sandbox.rollback()
            logger.error("Stage 6 exhausted all %d attempts.", 1 + max_retries)
            return FailureDiagnosis(
                stage="Stage 6: Patch Generator",
                reason=f"Exhausted maximum retry budget ({1 + max_retries} attempts) without passing tests.",
                baton=b3,
                details={"attempts": 1 + max_retries, "last_test_error": last_test_error},
            )
        finally:
            if hasattr(self.registry.sandbox, "cleanup"):
                try:
                    self.registry.sandbox.cleanup()
                except Exception as e:
                    logger.warning("Sandbox cleanup exception: %s", e)
