"""Inference engine and JSON constrained decoding for Qwen2.5-Coder-3B.

Handles prompt budgeting, ChatML formatting, grammar/JSON constrained decoding,
JSON repair with corrective retry loops, and usage tracking for Member 4.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Iterator, Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import AERSConfig, get_config
from src.model.llama_runtime import LlamaRuntime
from src.schemas import estimate_token_count

import threading
import psutil

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Qwen ChatML markers
CHATML_SYSTEM_START = "<|im_start|>system\n"
CHATML_USER_START = "<|im_start|>user\n"
CHATML_ASSISTANT_START = "<|im_start|>assistant\n"
CHATML_END = "<|im_end|>\n"
DEFAULT_STOP_TOKENS = ["<|im_end|>", "<|endoftext|>", "</s>"]


class UsageRecord(BaseModel):
    """Token and latency metrics for a single inference call."""
    stage: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    timestamp: float


class UsageTracker:
    """Tracks token consumption, latency metrics, and sampled peak resident RAM (RSS)."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []
        self.peak_rss_mb: float = self.get_current_rss_mb()
        self._monitor_thread: threading.Thread | None = None
        self._stop_monitor: threading.Event = threading.Event()

    @staticmethod
    def get_current_rss_mb() -> float:
        """Sample current process resident set size (RSS) in megabytes."""
        try:
            return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def start_memory_monitor(self, interval_sec: float = 0.05) -> None:
        """Start a background sampling thread to continuously measure peak RSS."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_monitor.clear()
        self.peak_rss_mb = max(self.peak_rss_mb, self.get_current_rss_mb())

        def _sample_loop() -> None:
            proc = psutil.Process()
            while not self._stop_monitor.is_set():
                try:
                    rss = proc.memory_info().rss / (1024 * 1024)
                    if rss > self.peak_rss_mb:
                        self.peak_rss_mb = round(rss, 2)
                except Exception:
                    pass
                time.sleep(interval_sec)

        self._monitor_thread = threading.Thread(target=_sample_loop, daemon=True, name="RSSMonitor")
        self._monitor_thread.start()
        logger.debug("Started background RSS memory monitor (sampling every %.2fs)", interval_sec)

    def stop_memory_monitor(self) -> float:
        """Stop background memory monitor and return peak observed RSS in MB."""
        self._stop_monitor.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
        self.peak_rss_mb = max(self.peak_rss_mb, self.get_current_rss_mb())
        logger.debug("Stopped memory monitor. Peak RSS: %.2f MB", self.peak_rss_mb)
        return self.peak_rss_mb

    def record(
        self,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> None:
        # Update peak RSS upon call completion
        self.peak_rss_mb = max(self.peak_rss_mb, self.get_current_rss_mb())
        rec = UsageRecord(
            stage=stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )
        self.records.append(rec)
        logger.debug(
            "Stage '%s' used %d prompt / %d completion tokens (%.1f ms) [Current RSS: %.1f MB]",
            stage,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            self.get_current_rss_mb(),
        )

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.records)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    def summary(self) -> dict[str, Any]:
        """Summary dictionary of usage metrics including measured peak RSS."""
        self.peak_rss_mb = max(self.peak_rss_mb, self.get_current_rss_mb())
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "stage_breakdown": {
                r.stage: {
                    "prompt_tokens": sum(x.prompt_tokens for x in self.records if x.stage == r.stage),
                    "completion_tokens": sum(x.completion_tokens for x in self.records if x.stage == r.stage),
                    "latency_ms": round(sum(x.latency_ms for x in self.records if x.stage == r.stage), 1),
                }
                for r in self.records
            },
        }

    def reset(self) -> None:
        self.records.clear()
        self.peak_rss_mb = self.get_current_rss_mb()


def repair_json_string(raw: str) -> str:
    """Repairs common LLM JSON syntax defects: markdown fences, trailing commas, single quotes."""
    text = raw.strip()

    # Strip markdown code blocks: ```json ... ``` or ``` ... ```
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
        else:
            # Drop triple backticks
            text = re.sub(r"^```(?:json)?", "", text)
            text = re.sub(r"```$", "", text).strip()

    # Find outermost JSON boundary
    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        end_brace = text.rfind("}")
        if end_brace != -1 and end_brace > start_brace:
            text = text[start_brace : end_brace + 1]
    elif start_bracket != -1:
        end_bracket = text.rfind("]")
        if end_bracket != -1 and end_bracket > start_bracket:
            text = text[start_bracket : end_bracket + 1]

    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([\]}])", r"\1", text)

    return text


def format_chatml_prompt(system: str, user: str) -> str:
    """Formats system and user messages into Qwen ChatML standard string."""
    return (
        f"{CHATML_SYSTEM_START}{system}{CHATML_END}"
        f"{CHATML_USER_START}{user}{CHATML_END}"
        f"{CHATML_ASSISTANT_START}"
    )


class InferenceEngine:
    """Shared inference engine backed by the singleton LlamaRuntime."""

    def __init__(
        self,
        runtime: LlamaRuntime | None = None,
        config: AERSConfig | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        self.config = config or get_config()
        self.runtime = runtime or LlamaRuntime.get_instance(self.config)
        self.tracker = tracker or UsageTracker()

    def count_tokens(self, text: str) -> int:
        """Count tokens using runtime tokenizer or fallback estimator."""
        if hasattr(self, "runtime") and self.runtime is not None:
            return self.runtime.count_tokens(text)
        return estimate_token_count(text)

    def _guard_and_truncate_prompt(
        self,
        system: str,
        user: str,
        max_tokens: int,
    ) -> str:
        """Ensures prompt does not exceed n_ctx - max_tokens, truncating user input if necessary."""
        n_ctx = self.config.n_ctx
        allowed_prompt = n_ctx - max_tokens

        system_tokens = self.count_tokens(format_chatml_prompt(system, ""))
        if system_tokens >= allowed_prompt:
            raise ValueError(
                f"System prompt ({system_tokens} tokens) exceeds allowed context window ({allowed_prompt} tokens)."
            )

        full_prompt = format_chatml_prompt(system, user)
        prompt_tokens = self.count_tokens(full_prompt)

        if prompt_tokens <= allowed_prompt:
            return full_prompt

        logger.warning(
            "Prompt exceeds budget (%d > %d tokens). Truncating user context to fit context window.",
            prompt_tokens,
            allowed_prompt,
        )

        # Budget remaining for user text
        budget_for_user_tokens = allowed_prompt - system_tokens - 10
        # Binary search / step truncation of user message
        approx_chars = int(budget_for_user_tokens * 3.5)
        truncated_user = user[:approx_chars] + "\n[... truncated to fit context budget ...]"
        return format_chatml_prompt(system, truncated_user)

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        stage_name: str = "general",
        grammar: Any | None = None,
    ) -> str:
        """Non-streaming text generation."""
        max_toks = max_tokens or 1024
        temp = temperature if temperature is not None else self.config.temperature
        stop_tokens = list(set((stop or []) + DEFAULT_STOP_TOKENS))

        prompt = self._guard_and_truncate_prompt(system, user, max_toks)
        prompt_token_count = self.runtime.count_tokens(prompt)

        start_time = time.perf_counter()

        if not self.runtime.is_loaded():
            raise RuntimeError(
                "Llama model is not loaded. Cannot generate without GGUF weights or FakeLLM."
            )

        with self.runtime._inference_lock:
            output = self.runtime.model(
                prompt=prompt,
                max_tokens=max_toks,
                temperature=temp,
                top_p=self.config.top_p,
                stop=stop_tokens,
                echo=False,
                grammar=grammar,
                seed=self.config.seed,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        text = output["choices"][0]["text"].strip()
        comp_token_count = output["usage"].get("completion_tokens", self.runtime.count_tokens(text))

        self.tracker.record(
            stage=stage_name,
            prompt_tokens=prompt_token_count,
            completion_tokens=comp_token_count,
            latency_ms=latency_ms,
        )
        self.last_call_metadata = {
            "stage": stage_name,
            "raw_output": text,
            "prompt_tokens": prompt_token_count,
            "completion_tokens": comp_token_count,
            "latency_ms": round(latency_ms, 2),
            "attempts": 1,
            "first_try_valid": True,
        }
        return text

    def stream(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        stage_name: str = "general",
    ) -> Iterator[str]:
        """Streaming token generation."""
        max_toks = max_tokens or 1024
        temp = temperature if temperature is not None else self.config.temperature
        stop_tokens = list(set((stop or []) + DEFAULT_STOP_TOKENS))

        prompt = self._guard_and_truncate_prompt(system, user, max_toks)
        prompt_token_count = self.runtime.count_tokens(prompt)

        start_time = time.perf_counter()

        if not self.runtime.is_loaded():
            raise RuntimeError("Llama model is not loaded for streaming.")

        with self.runtime._inference_lock:
            stream_gen = self.runtime.model(
                prompt=prompt,
                max_tokens=max_toks,
                temperature=temp,
                top_p=self.config.top_p,
                stop=stop_tokens,
                stream=True,
            )
            accumulated = []
            for chunk in stream_gen:
                delta = chunk["choices"][0].get("text", "")
                accumulated.append(delta)
                yield delta

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        full_text = "".join(accumulated)
        comp_token_count = self.runtime.count_tokens(full_text)
        self.tracker.record(
            stage=stage_name,
            prompt_tokens=prompt_token_count,
            completion_tokens=comp_token_count,
            latency_ms=latency_ms,
        )

    def generate_json(
        self,
        system: str,
        user: str,
        schema: Type[T] | dict[str, Any],
        max_tokens: int | None = None,
        max_retries: int = 2,
        stage_name: str = "json_stage",
    ) -> T | dict[str, Any]:
        """Generates JSON validated against a Pydantic schema or dict schema.
        
        Attempts constrained decoding first via LlamaGrammar; repairs and retries
        with corrective prompts up to max_retries if needed.
        """
        curr_user = user
        last_error = ""
        last_output = ""

        # Check if schema is Pydantic class or dict
        is_pydantic = isinstance(schema, type) and issubclass(schema, BaseModel)

        schema_hint = (
            schema.model_json_schema() if is_pydantic else schema
        )
        augmented_system = (
            f"{system}\n\nSTRICT JSON OUTPUT REQUIREMENT:\n"
            f"You MUST output valid raw JSON adhering to this JSON Schema:\n"
            f"{json.dumps(schema_hint, indent=2)}\n"
            f"Do not include explanation, chain of thought, or markdown fences."
        )

        # Compile grammar for constrained decoding if available
        grammar = None
        if not self.config.dry_run and self.runtime.is_loaded():
            try:
                from llama_cpp.llama_grammar import LlamaGrammar
                schema_json_str = json.dumps(schema_hint)
                grammar = LlamaGrammar.from_json_schema(schema_json_str)
                logger.debug("Successfully compiled LlamaGrammar for %s", stage_name)
            except Exception as e:
                logger.warning("Could not compile LlamaGrammar for %s: %s", stage_name, e)
                grammar = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.info(
                    "Retrying JSON generation (attempt %d/%d) for stage %s due to error: %s",
                    attempt,
                    max_retries,
                    stage_name,
                    last_error,
                )
                curr_user = (
                    f"{user}\n\n"
                    f"CRITICAL CORRECTION (Attempt {attempt}):\n"
                    f"Your previous output failed validation with error: {last_error}\n"
                    f"Your previous output was:\n{last_output[:400]}\n"
                    f"Output ONLY valid JSON matching the schema."
                )

            raw = self.generate(
                system=augmented_system,
                user=curr_user,
                max_tokens=max_tokens,
                stage_name=stage_name,
                grammar=grammar,
            )
            last_output = raw

            # Clean and repair common syntax issues as fallback
            repaired = repair_json_string(raw)

            try:
                parsed_dict = json.loads(repaired)
                if is_pydantic:
                    result = schema.model_validate(parsed_dict)
                else:
                    result = parsed_dict

                # Record metadata for reporting and observability
                prev_meta = getattr(self, "last_call_metadata", {})
                self.last_call_metadata = {
                    "stage": stage_name,
                    "raw_output": raw,
                    "repaired_output": repaired,
                    "attempts": attempt + 1,
                    "first_try_valid": (attempt == 0 and repaired == raw.strip()),
                    "needed_repair": (repaired != raw.strip()),
                    "grammar_constrained": (grammar is not None),
                    "prompt_tokens": prev_meta.get("prompt_tokens", 0),
                    "completion_tokens": prev_meta.get("completion_tokens", 0),
                    "latency_ms": prev_meta.get("latency_ms", 0.0),
                }
                return result
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                last_error = str(e)
                logger.warning("JSON parsing/validation failed on attempt %d: %s", attempt, e)

        raise ValueError(
            f"Failed to generate valid JSON for stage '{stage_name}' after {max_retries} retries. "
            f"Last error: {last_error}\nLast output:\n{last_output}"
        )


class FakeLLM(InferenceEngine):
    """Mock inference engine for fast testing without loading GGUF weights."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.config = get_config()
        self.runtime = None
        self.tracker = UsageTracker()
        self.responses: dict[str, Any] = responses or {}
        self.calls: list[dict[str, Any]] = []

    def set_response(self, stage_name: str, response: Any) -> None:
        """Register canned response for a stage."""
        self.responses[stage_name] = response

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        stage_name: str = "general",
    ) -> str:
        self.calls.append({"stage": stage_name, "system": system, "user": user})
        self.tracker.record(stage=stage_name, prompt_tokens=150, completion_tokens=100, latency_ms=10.0)

        if stage_name in self.responses:
            resp = self.responses[stage_name]
            if isinstance(resp, str):
                return resp
            if isinstance(resp, BaseModel):
                return resp.model_dump_json()
            return json.dumps(resp)

        # Default mock responses
        if "patch" in stage_name.lower():
            return (
                "--- a/src/target.py\n"
                "+++ b/src/target.py\n"
                "@@ -10,3 +10,3 @@\n"
                "-    return a - b\n"
                "+    return a + b\n"
            )
        return "Mock LLM output"

    def generate_json(
        self,
        system: str,
        user: str,
        schema: Type[T] | dict[str, Any],
        max_tokens: int | None = None,
        max_retries: int = 2,
        stage_name: str = "json_stage",
    ) -> T | dict[str, Any]:
        self.calls.append({"stage": stage_name, "system": system, "user": user})
        self.tracker.record(stage=stage_name, prompt_tokens=200, completion_tokens=80, latency_ms=12.0)

        is_pydantic = isinstance(schema, type) and issubclass(schema, BaseModel)

        if stage_name in self.responses:
            resp = self.responses[stage_name]
            if is_pydantic:
                if isinstance(resp, schema):
                    return resp
                if isinstance(resp, dict):
                    return schema.model_validate(resp)
                if isinstance(resp, str):
                    repaired = repair_json_string(resp)
                    return schema.model_validate_json(repaired)
            elif isinstance(resp, dict):
                return resp

        # Default fallback instance for Pydantic schema
        if is_pydantic:
            try:
                # Try creating default instance with dummy values
                from src.schemas import Hypothesis, LocalizationResult, RequirementContract, SuspiciousRange
                if issubclass(schema, RequirementContract):
                    return schema(
                        task_type="bug_fix",
                        observed_behavior="Defect observed in execution",
                        expected_behavior="Expected correct execution",
                        inviolable_constraints=["Do not break backwards compatibility"],
                        acceptance_oracles=["test_bug_reproduction"],
                    )
                if issubclass(schema, LocalizationResult):
                    return schema(
                        candidate_files=["src/target.py"],
                        suspicious_ranges=[SuspiciousRange(file="src/target.py", start=10, end=15, confidence=0.9)],
                    )
                if issubclass(schema, Hypothesis):
                    return schema(
                        id="H1",
                        statement="Sign inversion in arithmetic operation",
                        failure_mechanism="Subtraction used instead of addition",
                        falsification_test="Assert positive sum with positive inputs",
                        status="PROPOSED",
                        score=0.9,
                    )
            except Exception as e:
                logger.error("Failed to construct default schema instance: %s", e)

        return {}
