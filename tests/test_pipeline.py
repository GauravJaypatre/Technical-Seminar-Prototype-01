"""End-to-end tests for AERSPipeline."""

from pathlib import Path
import pytest

from src.config import AERSConfig
from src.model.inference import FakeLLM
from src.pipeline import AERSPipeline
from src.registry import ComponentRegistry
from src.schemas import FailureDiagnosis, PatchResult
from src.stubs.test_runner_stub import TestRunnerStub


def test_pipeline_success_first_attempt(sample_repo: Path, mock_config: AERSConfig):
    """Verify pipeline completes end-to-end and returns PatchResult on first attempt."""
    registry = ComponentRegistry()
    registry.test_runner = TestRunnerStub(force_pass=True)

    pipeline = AERSPipeline(
        config=mock_config,
        registry=registry,
    )

    issue = "Fix division by zero when b=0 in divide(a, b)"
    result = pipeline.run(issue=issue, repo_path=sample_repo)

    assert isinstance(result, PatchResult)
    assert result.passed is True
    assert result.attempts == 1
    assert "---" in result.diff


def test_pipeline_success_after_retries(sample_repo: Path, mock_config: AERSConfig):
    """Verify pipeline retries when test fails and succeeds on attempt 2."""
    registry = ComponentRegistry()
    # Fails attempt 1, passes attempt 2
    registry.test_runner = TestRunnerStub(fail_attempts_until=1)

    pipeline = AERSPipeline(
        config=mock_config,
        registry=registry,
    )

    issue = "Fix division by zero when b=0 in divide(a, b)"
    result = pipeline.run(issue=issue, repo_path=sample_repo)

    assert isinstance(result, PatchResult)
    assert result.passed is True
    assert result.attempts == 2


def test_pipeline_exhausted_retries_returns_failure(sample_repo: Path, mock_config: AERSConfig):
    """Verify pipeline returns FailureDiagnosis when all retries are exhausted."""
    registry = ComponentRegistry()
    # Always fail tests
    registry.test_runner = TestRunnerStub(force_pass=False)

    pipeline = AERSPipeline(
        config=mock_config,
        registry=registry,
    )

    issue = "Fix division by zero when b=0 in divide(a, b)"
    result = pipeline.run(issue=issue, repo_path=sample_repo)

    assert isinstance(result, FailureDiagnosis)
    assert result.stage == "Stage 6: Patch Generator"
    assert "Exhausted maximum retry budget" in result.reason
    assert result.baton.token_count() <= mock_config.baton_max_tokens
    assert len(result.baton.retry_traces) > 0


def test_pipeline_structured_logging(sample_repo: Path, mock_config: AERSConfig, tmp_path: Path):
    """Verify pipeline writes structured events to JSONL."""
    log_dir = tmp_path / "test_logs"
    mock_config.log_dir = str(log_dir)

    registry = ComponentRegistry()
    registry.test_runner = TestRunnerStub(force_pass=True)

    pipeline = AERSPipeline(
        config=mock_config,
        registry=registry,
    )

    issue = "Fix division by zero"
    pipeline.run(issue=issue, repo_path=sample_repo)

    log_files = list(log_dir.glob("pipeline_*.jsonl"))
    assert len(log_files) == 1
    lines = log_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 6  # Stages 1 to 6 logged


@pytest.mark.slow
def test_real_model_optional():
    """Optional slow test using the real GGUF model if present on disk."""
    model_path = Path("models/qwen2.5-coder-3b-instruct-q4_k_m.gguf")
    if not model_path.exists():
        pytest.skip("GGUF model file not found. Skipping slow model inference test.")

    from src.model.llama_runtime import LlamaRuntime
    runtime = LlamaRuntime.get_instance()
    assert runtime.is_loaded()
    tokens = runtime.tokenize("def hello(): pass")
    assert len(tokens) > 0
