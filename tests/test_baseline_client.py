"""
Unit tests for the provider-agnostic baseline client abstraction,
rate limiting, exponential backoff, and prototype tagging.
"""
import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.baseline_client import (
    GenerationResult,
    BaseLLMBackend,
    AnthropicBackend,
    OpenAICompatBackend,
    RPMThrottle,
    get_baseline_client
)
from src.benchmarking_extended import BenchmarkLogger
from src.sandbox import clean_diff_text


def test_generation_result_unpacking():
    """Verify GenerationResult unpacks into a 4-tuple for backward compatibility."""
    res = GenerationResult(
        text="--- a/test.py\n+++ b/test.py",
        prompt_tokens=150,
        completion_tokens=42,
        latency_sec=1.23,
        provider="groq",
        model_name="openai/gpt-oss-120b"
    )
    # Test unpacking
    text, p_tok, c_tok, lat = res
    assert text == "--- a/test.py\n+++ b/test.py"
    assert p_tok == 150
    assert c_tok == 42
    assert lat == 1.23
    assert res.provider == "groq"
    assert res.model_name == "openai/gpt-oss-120b"


def test_client_factory_defaults_to_anthropic(monkeypatch):
    """Verify factory defaults strictly to anthropic if BASELINE_PROVIDER is unset."""
    monkeypatch.delenv("BASELINE_PROVIDER", raising=False)
    monkeypatch.delenv("BASELINE_API_KEY", raising=False)
    client = get_baseline_client()
    assert isinstance(client, AnthropicBackend)
    assert client.provider == "anthropic"


def test_client_factory_selects_groq(monkeypatch):
    """Verify factory instantiates OpenAICompatBackend when BASELINE_PROVIDER=groq."""
    monkeypatch.setenv("BASELINE_PROVIDER", "groq")
    monkeypatch.setenv("BASELINE_API_KEY", "gsk_test_key_123")
    monkeypatch.delenv("BASELINE_BASE_URL", raising=False)
    client = get_baseline_client()
    assert isinstance(client, OpenAICompatBackend)
    assert client.provider == "groq"
    assert client.base_url == "https://api.groq.com/openai/v1"


def test_client_factory_selects_openrouter(monkeypatch):
    """Verify factory instantiates OpenAICompatBackend when BASELINE_PROVIDER=openrouter."""
    monkeypatch.setenv("BASELINE_PROVIDER", "openrouter")
    monkeypatch.setenv("BASELINE_API_KEY", "sk-or-v1-test-key")
    monkeypatch.delenv("BASELINE_BASE_URL", raising=False)
    client = get_baseline_client()
    assert isinstance(client, OpenAICompatBackend)
    assert client.provider == "openrouter"
    assert client.base_url == "https://openrouter.ai/api/v1"



def test_rpm_throttle_timing():
    """Verify RPMThrottle enforces delay between sequential calls."""
    # 600 RPM = 0.1s minimum interval
    throttle = RPMThrottle(max_rpm=600)
    throttle.wait_if_needed()

    start = time.time()
    throttle.wait_if_needed()
    elapsed = time.time() - start
    assert elapsed >= 0.08  # Tolerates minor clock jitter


def test_openai_compat_backend_success():
    """Verify OpenAICompatBackend parses standard OpenAI response."""
    backend = OpenAICompatBackend(
        provider="groq",
        api_key="gsk_test_key_123",
        base_url="https://api.groq.com/openai/v1",
        model_name="openai/gpt-oss-120b",
        max_rpm=0
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "```diff\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n```"
            }
        }],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 35
        }
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = backend.generate("Fix the bug")
        assert "--- a/file.py" in result.text
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 35
        assert result.provider == "groq"
        assert result.model_name == "openai/gpt-oss-120b"
        mock_post.assert_called_once()


def test_openai_compat_backend_exponential_backoff_on_429():
    """Verify OpenAICompatBackend retries on 429 with backoff and succeeds."""
    backend = OpenAICompatBackend(
        provider="groq",
        api_key="gsk_test_key_123",
        base_url="https://api.groq.com/openai/v1",
        model_name="openai/gpt-oss-120b",
        max_rpm=0,
        max_retries=3
    )

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0.1"}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [{"message": {"content": "diff content"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10}
    }

    with patch("requests.post", side_effect=[resp_429, resp_200]):
        result = backend.generate("Test prompt")
        assert result.text == "diff content"
        assert result.prompt_tokens == 50


def test_prototype_logging_isolation(tmp_path):
    """Verify BenchmarkLogger writes to prototype directory with unambiguous tags."""
    proto_dir = str(tmp_path / "prototype_run")
    logger = BenchmarkLogger(
        session_name="test_proto",
        output_dir=proto_dir,
        is_prototype=True
    )

    record = {
        "run_id": "proto_test_01_s42",
        "timestamp": "2026-09-08T12:00:00",
        "task_id": "easy_01",
        "difficulty": "easy",
        "system": "monolithic_frontier",
        "model_profile": "openai/gpt-oss-120b",
        "provider": "groq",
        "model_name": "openai/gpt-oss-120b",
        "is_prototype": True,
        "seed": 42,
        "pass_at_1": True,
        "final_pass": True,
        "iterations_to_success": 1,
        "total_iterations_run": 1,
        "cumulative_wall_clock_sec": 1.45,
        "total_prompt_tokens": 500,
        "total_completion_tokens": 120,
        "estimated_cost_usd": 0.0,
        "peak_ram_mb": 45.2
    }
    logger.log_run(record)

    assert os.path.exists(logger.csv_path)
    assert os.path.exists(logger.json_path)

    with open(logger.csv_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "provider" in content
    assert "is_prototype" in content
    assert "groq" in content
    assert "openai/gpt-oss-120b" in content


def test_clean_diff_text_resilience():
    """Verify clean_diff_text handles various LLM markdown fences and raw diffs."""
    # Case 1: Standard ```diff fence
    text1 = "Here is the fix:\n```diff\n--- a/main.py\n+++ b/main.py\n@@ -1,1 +1,1 @@\n-x\n+y\n```"
    assert clean_diff_text(text1).startswith("--- a/main.py")

    # Case 2: ```patch fence
    text2 = "```patch\n--- a/mod.py\n+++ b/mod.py\n@@ -1,1 +1,1 @@\n-a\n+b\n```"
    assert clean_diff_text(text2).startswith("--- a/mod.py")

    # Case 3: Raw diff without fences
    text3 = "Explanation here.\n--- a/raw.py\n+++ b/raw.py\n@@ -1 +1 @@\n-1\n+2"
    assert clean_diff_text(text3).startswith("--- a/raw.py")


def test_provider_mismatch_guard_groq_rejects_openrouter_key():
    """Verify OpenAICompatBackend raises ValueError if provider=groq but key is OpenRouter."""
    with pytest.raises(ValueError, match="Provider mismatch guard.*sk-or-"):
        OpenAICompatBackend(
            provider="groq",
            api_key="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",
            base_url="https://api.groq.com/openai/v1"
        )


def test_provider_mismatch_guard_groq_rejects_openrouter_url():
    """Verify OpenAICompatBackend raises ValueError if provider=groq but base_url is OpenRouter."""
    with pytest.raises(ValueError, match="Provider mismatch guard.*OpenRouter"):
        OpenAICompatBackend(
            provider="groq",
            api_key="gsk_valid_key_123",
            base_url="https://openrouter.ai/api/v1"
        )


def test_provider_mismatch_guard_openrouter_rejects_groq_key():
    """Verify OpenAICompatBackend raises ValueError if provider=openrouter but key is Groq."""
    with pytest.raises(ValueError, match="Provider mismatch guard.*gsk_"):
        OpenAICompatBackend(
            provider="openrouter",
            api_key="gsk_test_key_123",
            base_url="https://openrouter.ai/api/v1"
        )


def test_provider_mismatch_guard_openrouter_rejects_groq_url():
    """Verify OpenAICompatBackend raises ValueError if provider=openrouter but base_url is Groq."""
    with pytest.raises(ValueError, match="Provider mismatch guard.*Groq"):
        OpenAICompatBackend(
            provider="openrouter",
            api_key="sk-or-v1-test-key",
            base_url="https://api.groq.com/openai/v1"
        )


def test_provider_mismatch_guard_anthropic_rejects_groq_and_openrouter_keys():
    """Verify AnthropicBackend raises ValueError if key is Groq or OpenRouter."""
    with pytest.raises(ValueError, match="Provider mismatch guard.*Groq"):
        AnthropicBackend(api_key="gsk_invalid_for_anthropic")

    with pytest.raises(ValueError, match="Provider mismatch guard.*OpenRouter"):
        AnthropicBackend(api_key="sk-or-v1-invalid-for-anthropic")


def test_provider_alignment_valid_groq_key_passes():
    """Verify valid Groq configuration initializes without error."""
    client = OpenAICompatBackend(
        provider="groq",
        api_key="gsk_valid_test_key_abc123",
        base_url="https://api.groq.com/openai/v1"
    )
    assert client.provider == "groq"
    assert client.throttle.max_rpm == 8
    assert client.throttle.max_tpm == 8000


def test_rpm_throttle_tpm_window_tracking():
    """Verify RPMThrottle tracks token consumption and enforces delays when headroom is low."""
    throttle = RPMThrottle(max_rpm=600, max_tpm=8000, safe_token_margin=1000)
    throttle.record_usage(total_tokens=1500, remaining_tokens=500, reset_sec=0.1)
    assert len(throttle.token_history) == 1
    assert throttle.last_remaining_tokens == 500

    start = time.time()
    throttle.wait_if_needed()
    elapsed = time.time() - start
    assert elapsed >= 0.4

