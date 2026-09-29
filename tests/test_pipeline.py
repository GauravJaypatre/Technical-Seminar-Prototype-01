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
    """Thorough test verifying real Qwen2.5-Coder-3B GGUF model bring-up.

    Tests:
    1. CPU-only configuration (n_gpu_layers=0).
    2. Singleton initialization (loaded exactly once in memory).
    3. Real generate() call and output validation.
    4. Real generate_json() call with RequirementContract schema validation.
    5. Peak RSS memory monitoring.
    """
    model_path = Path("models/qwen2.5-coder-3b-instruct-q4_k_m.gguf")
    if not model_path.exists():
        pytest.skip(f"GGUF model file not found at {model_path}. Skipping slow model test.")

    from src.config import AERSConfig
    from src.model.llama_runtime import LlamaRuntime
    from src.model.inference import InferenceEngine
    from src.schemas import RequirementContract

    config = AERSConfig(
        model_path=str(model_path),
        n_ctx=4096,
        n_gpu_layers=0,  # CPU-only
        dry_run=False,
    )

    # Verify Singleton behavior
    runtime1 = LlamaRuntime.get_instance(config)
    runtime2 = LlamaRuntime.get_instance(config)
    assert runtime1 is runtime2, "LlamaRuntime must be a singleton instance"
    assert runtime1.is_loaded(), "Model must be loaded into memory"
    assert runtime1.model is runtime2.model, "Underlying model must be shared"

    engine = InferenceEngine(runtime=runtime1, config=config)

    # 1. Test generate() non-streaming
    raw_output = engine.generate(
        system="You are a precise coding assistant.",
        user="Write a Python function called `add` that takes two numbers and returns their sum. Return code only.",
        max_tokens=150,
        stage_name="test_generate",
    )
    assert len(raw_output.strip()) > 0, "Model generate() output must be non-empty"
    assert "def add" in raw_output, f"Expected 'def add' in output, got: {raw_output}"

    # 2. Test generate_json() with Pydantic schema
    json_prompt = "The user reported: 'ZeroDivisionError in calculate_ratio when total is zero'. Extract the requirement contract."
    contract = engine.generate_json(
        system="You are an expert software analyst. Output raw JSON only adhering to the schema.",
        user=json_prompt,
        schema=RequirementContract,
        max_tokens=300,
        stage_name="test_generate_json",
    )
    assert isinstance(contract, RequirementContract), "Must return a validated RequirementContract"
    assert contract.task_type in {"bug_fix", "feature", "refactor"}
    assert len(contract.observed_behavior) > 0
    assert len(contract.expected_behavior) > 0

    # 3. Assert peak memory stayed within acceptable limits (< 6 GB)
    peak_rss = engine.tracker.peak_rss_mb
    assert peak_rss < 6144.0, f"Peak RSS memory ({peak_rss:.1f} MB) exceeded 6 GB"
