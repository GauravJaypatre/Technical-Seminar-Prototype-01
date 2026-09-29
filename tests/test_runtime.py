import json
import time

import pytest

from src.config import AERSConfig
from src.model.inference import (
    FakeLLM,
    InferenceEngine,
    UsageTracker,
    format_chatml_prompt,
    repair_json_string,
)
from src.schemas import RequirementContract


def test_repair_json_string_fences():
    """Verify stripping markdown code fences."""
    raw = "```json\n{\"task_type\": \"bug_fix\", \"observed_behavior\": \"err\"}\n```"
    repaired = repair_json_string(raw)
    assert repaired == '{"task_type": "bug_fix", "observed_behavior": "err"}'


def test_repair_json_string_trailing_commas():
    """Verify removing trailing commas from objects and lists."""
    raw = '{"items": [1, 2, 3,], "key": "val",}'
    repaired = repair_json_string(raw)
    assert repaired == '{"items": [1, 2, 3], "key": "val"}'


def test_repair_json_embedded_in_chatter():
    """Verify extracting JSON block when surrounded by prose."""
    raw = (
        "Here is the requested specification:\n"
        "{\n  \"task_type\": \"bug_fix\",\n  \"observed_behavior\": \"crash\"\n}\n"
        "Let me know if you need more details!"
    )
    repaired = repair_json_string(raw)
    parsed = json.loads(repaired)
    assert parsed["task_type"] == "bug_fix"
    assert parsed["observed_behavior"] == "crash"
    assert not repaired.startswith("Here is")
    assert not repaired.endswith("details!")


def test_usage_tracker_metrics():
    """Verify UsageTracker aggregates totals and stage breakdowns correctly."""
    tracker = UsageTracker()
    tracker.record("stage1", prompt_tokens=100, completion_tokens=50, latency_ms=120.0)
    tracker.record("stage3", prompt_tokens=200, completion_tokens=80, latency_ms=250.0)

    assert tracker.total_prompt_tokens == 300
    assert tracker.total_completion_tokens == 130
    assert tracker.total_tokens == 430
    assert tracker.total_calls == 2
    assert tracker.total_latency_ms == 370.0

    summary = tracker.summary()
    assert summary["total_calls"] == 2
    assert "stage1" in summary["stage_breakdown"]
    assert summary["stage_breakdown"]["stage1"]["prompt_tokens"] == 100
    assert summary["peak_rss_mb"] > 0.0


def test_usage_tracker_memory_monitor():
    """Verify background RSS memory monitor samples resident set size."""
    tracker = UsageTracker()
    tracker.start_memory_monitor(interval_sec=0.01)
    # Allocate a small buffer
    _ = [0] * 500_000
    time.sleep(0.05)
    peak = tracker.stop_memory_monitor()
    assert peak > 0.0
    assert tracker.peak_rss_mb == peak


def test_guard_and_truncate_prompt():
    """Verify prompt truncation guard enforces context window."""
    cfg = AERSConfig(n_ctx=100, max_tokens_profiler=20)
    engine = FakeLLM()
    engine.config = cfg

    # Normal prompt
    normal = engine._guard_and_truncate_prompt("system prompt", "user query", max_tokens=20)
    assert "user query" in normal

    # Excessive user prompt
    huge_user = "word " * 500
    guarded = engine._guard_and_truncate_prompt("system prompt", huge_user, max_tokens=20)
    assert "truncated to fit context budget" in guarded


def test_guard_raises_if_system_prompt_exceeds_budget():
    """Verify ValueError is raised if system prompt alone is too large."""
    cfg = AERSConfig(n_ctx=50)
    engine = FakeLLM()
    engine.config = cfg

    huge_system = "system " * 200
    with pytest.raises(ValueError, match="exceeds allowed context window"):
        engine._guard_and_truncate_prompt(huge_system, "short user", max_tokens=30)


def test_chatml_formatting():
    """Verify Qwen ChatML prompt format."""
    formatted = format_chatml_prompt("sys", "usr")
    assert formatted.startswith("<|im_start|>system\nsys<|im_end|>\n")
    assert "<|im_start|>user\nusr<|im_end|>\n" in formatted
    assert formatted.endswith("<|im_start|>assistant\n")
