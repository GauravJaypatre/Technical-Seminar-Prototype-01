"""Pydantic schemas and dataclasses for AERS-SLM.

Defines the core data contracts between pipeline stages, including the
EvidenceBaton (inter-stage communication state with hard token budgets)
and the RequirementContract.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Default token budget constants
INITIAL_BATON_MAX_TOKENS = 500
BATON_HARD_CAP_TOKENS = 1200
MAX_CANDIDATE_FILES = 5
MAX_HYPOTHESES = 3


def estimate_token_count(text: str) -> int:
    """Fast, conservative BPE-style token count estimator.
    
    If tiktoken or llama_cpp tokenizer is not available, uses an empirical
    regex-based subword / character-ratio estimator (~3.6 chars per token
    with punctuation/whitespace weighting).
    """
    if not text:
        return 0
    # Split on whitespace and punctuation tokens
    tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
    # Accounting for subwords: roughly average 1 token per 3.7 characters for code/json
    char_estimate = math.ceil(len(text) / 3.7)
    return max(len(tokens), char_estimate)


class RequirementContract(BaseModel):
    """Stage 1 contract: Formal specification extracted from the issue and repo profile.
    
    Contains EXACTLY the five required fields.
    """
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["bug_fix", "feature", "refactor"] = Field(
        default="bug_fix",
        description="Type of task: bug_fix, feature, or refactor."
    )
    observed_behavior: str = Field(
        ...,
        description="Empirical description of current incorrect behavior or defect."
    )
    expected_behavior: str = Field(
        ...,
        description="Prescriptive description of desired correct behavior."
    )
    inviolable_constraints: list[str] = Field(
        default_factory=list,
        description="Hard constraints that any valid solution must never violate."
    )
    acceptance_oracles: list[str] = Field(
        default_factory=list,
        description="Testable assertions or test functions that verify correctness."
    )

    @field_validator("task_type", mode="before")
    @classmethod
    def validate_task_type(cls, v: Any) -> str:
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in {"bug_fix", "feature", "refactor"}:
                return clean
            # Fallback if model outputs slight variation
            if "bug" in clean or "fix" in clean:
                return "bug_fix"
            if "feat" in clean:
                return "feature"
            if "refactor" in clean:
                return "refactor"
        return "bug_fix"


class SuspiciousRange(BaseModel):
    """Suspicious file and line range identified during fault localization."""
    file: str
    start: int
    end: int
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class LocalizationResult(BaseModel):
    """Output of Stage 3 Fault Localizer."""
    candidate_files: list[str] = Field(default_factory=list)
    suspicious_ranges: list[SuspiciousRange] = Field(default_factory=list)

    @field_validator("candidate_files")
    @classmethod
    def cap_candidate_files(cls, v: list[str]) -> list[str]:
        return v[:MAX_CANDIDATE_FILES]


class Hypothesis(BaseModel):
    """Stage 4 root-cause hypothesis with empirical falsification condition."""
    id: str
    statement: str
    failure_mechanism: str
    falsification_test: str
    status: str = Field(
        default="PROPOSED",
        description="Hypothesis state: PROPOSED, SUPPORTED, FALSIFIED, or REJECTED."
    )
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceBaton(BaseModel):
    """The bounded, typed JSON evidence baton passed sequentially across stages.
    
    Hard token budget is strictly <= 1,200 tokens.
    Initial baton B0 must be <= 500 tokens.
    Conversational history and unverified reasoning are strictly excluded.
    """
    requirement: RequirementContract
    candidate_files: list[str] = Field(default_factory=list)
    suspicious_ranges: list[SuspiciousRange] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    verified_hypothesis: Hypothesis | None = None
    retry_traces: list[str] = Field(
        default_factory=list,
        description="Last error traceback only, truncated to fit token budget."
    )
    confidence_scores: dict[str, float] = Field(default_factory=dict)

    def to_json(self, indent: int | None = None) -> str:
        """Serialize baton to standard JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> EvidenceBaton:
        """Deserialize baton from JSON string."""
        return cls.model_validate_json(data)

    def token_count(self, tokenizer: Callable[[str], int] | None = None) -> int:
        """Count tokens of the serialized baton."""
        text = self.to_json()
        if tokenizer is not None:
            return tokenizer(text)
        return estimate_token_count(text)

    def project(
        self,
        max_tokens: int = BATON_HARD_CAP_TOKENS,
        tokenizer: Callable[[str], int] | None = None
    ) -> EvidenceBaton:
        """Enforces token caps by dropping or truncating lower-priority fields.
        
        Priority order (highest to lowest preserved):
        1. RequirementContract (inviolable core specifications)
        2. Verified hypothesis (if present)
        3. Top suspicious ranges (max 5) & candidate files (max 5)
        4. Hypotheses (max 3)
        5. Retry traces (only last error, progressively truncated)
        
        Raises ValueError if even the minimal core cannot fit within max_tokens.
        """
        # Step 1: Enforce structural caps
        capped_files = self.candidate_files[:MAX_CANDIDATE_FILES]
        capped_ranges = sorted(
            self.suspicious_ranges,
            key=lambda r: r.confidence,
            reverse=True
        )[:MAX_CANDIDATE_FILES]
        capped_hypotheses = self.hypotheses[:MAX_HYPOTHESES]
        
        # Step 2: Keep only the most recent error trace (if any)
        capped_traces = [self.retry_traces[-1]] if self.retry_traces else []

        projected = EvidenceBaton(
            requirement=self.requirement,
            candidate_files=capped_files,
            suspicious_ranges=capped_ranges,
            hypotheses=capped_hypotheses,
            verified_hypothesis=self.verified_hypothesis,
            retry_traces=capped_traces,
            confidence_scores=self.confidence_scores,
        )

        current_tokens = projected.token_count(tokenizer)
        if current_tokens <= max_tokens:
            return projected

        # Step 3: Progressively compress retry_traces if still exceeding
        if projected.retry_traces:
            trace = projected.retry_traces[0]
            # Truncate trace in steps (500 chars -> 250 chars -> 100 chars -> dropped)
            for char_limit in (400, 200, 100):
                if len(trace) > char_limit:
                    projected.retry_traces = [trace[-char_limit:] + "\n[truncated]"]
                    if projected.token_count(tokenizer) <= max_tokens:
                        return projected
            # Drop trace entirely if still over budget
            projected.retry_traces = []
            if projected.token_count(tokenizer) <= max_tokens:
                return projected

        # Step 4: Drop non-verified hypotheses if still over budget
        if not projected.verified_hypothesis and len(projected.hypotheses) > 1:
            projected.hypotheses = projected.hypotheses[:1]
            if projected.token_count(tokenizer) <= max_tokens:
                return projected

        # Step 5: Trim suspicious ranges to top 2
        if len(projected.suspicious_ranges) > 2:
            projected.suspicious_ranges = projected.suspicious_ranges[:2]
            if projected.token_count(tokenizer) <= max_tokens:
                return projected

        final_tokens = projected.token_count(tokenizer)
        if final_tokens > max_tokens:
            msg = (
                f"EvidenceBaton could not be projected within {max_tokens} tokens! "
                f"Current size: {final_tokens} tokens."
            )
            logger.error(msg)
            raise ValueError(msg)

        return projected

    def update(
        self,
        candidate_files: list[str] | None = None,
        suspicious_ranges: list[SuspiciousRange] | None = None,
        hypotheses: list[Hypothesis] | None = None,
        verified_hypothesis: Hypothesis | None = None,
        confidence_scores: dict[str, float] | None = None,
        tokenizer: Callable[[str], int] | None = None,
    ) -> EvidenceBaton:
        """Create a new EvidenceBaton with updated fields and enforce projection."""
        new_data = self.model_dump()
        if candidate_files is not None:
            new_data["candidate_files"] = candidate_files[:MAX_CANDIDATE_FILES]
        if suspicious_ranges is not None:
            new_data["suspicious_ranges"] = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in suspicious_ranges
            ]
        if hypotheses is not None:
            new_data["hypotheses"] = [
                h.model_dump() if hasattr(h, "model_dump") else h
                for h in hypotheses[:MAX_HYPOTHESES]
            ]
        if verified_hypothesis is not None:
            new_data["verified_hypothesis"] = (
                verified_hypothesis.model_dump()
                if hasattr(verified_hypothesis, "model_dump")
                else verified_hypothesis
            )
        if confidence_scores is not None:
            new_data["confidence_scores"] = {
                **self.confidence_scores,
                **confidence_scores
            }

        updated = EvidenceBaton.model_validate(new_data)
        return updated.project(max_tokens=BATON_HARD_CAP_TOKENS, tokenizer=tokenizer)

    def append_traceback(
        self,
        error_log: str,
        max_chars: int = 500,
        tokenizer: Callable[[str], int] | None = None
    ) -> EvidenceBaton:
        """Appends the most recent failure traceback (truncated) and drops older traces."""
        clean_log = error_log.strip()
        if len(clean_log) > max_chars:
            clean_log = clean_log[-max_chars:] + "\n[truncated]"
        new_data = self.model_dump()
        new_data["retry_traces"] = [clean_log]
        updated = EvidenceBaton.model_validate(new_data)
        return updated.project(max_tokens=BATON_HARD_CAP_TOKENS, tokenizer=tokenizer)


class PatchResult(BaseModel):
    """Result of Stage 6 Patch Generator & Test Validation."""
    diff: str = Field(..., description="Unified diff string.")
    passed: bool = Field(..., description="Whether test suite passed with patch.")
    attempts: int = Field(default=1, description="Number of patch generation attempts.")
    test_log: str = Field(default="", description="Output logs from test execution.")


class FailureDiagnosis(BaseModel):
    """Structured failure diagnosis returned when the pipeline cannot produce a fix."""
    stage: str = Field(..., description="Name of the stage where failure occurred.")
    reason: str = Field(..., description="Explanation of the failure cause.")
    baton: EvidenceBaton = Field(..., description="State of the EvidenceBaton at failure.")
    details: dict[str, Any] = Field(default_factory=dict)
