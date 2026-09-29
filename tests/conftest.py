"""Pytest shared fixtures for AERS-SLM test suite."""

from pathlib import Path
import shutil
import tempfile
import pytest

from src.config import AERSConfig
from src.model.inference import FakeLLM
from src.schemas import RequirementContract, SuspiciousRange


@pytest.fixture
def mock_config() -> AERSConfig:
    """Provides a deterministic test configuration with dry_run enabled."""
    return AERSConfig(
        model_path="models/test_model.gguf",
        n_ctx=4096,
        n_threads=2,
        temperature=0.2,
        dry_run=True,
        context_budget=2000,
        baton_max_tokens=1200,
        initial_baton_max_tokens=500,
        max_retries=2,
    )


@pytest.fixture
def fake_engine(mock_config: AERSConfig) -> FakeLLM:
    """Provides an instrumented FakeLLM engine."""
    engine = FakeLLM()
    engine.config = mock_config
    return engine


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Creates a temporary sample repository with real python files for testing."""
    repo = tmp_path / "sample_project"
    repo.mkdir()
    
    src = repo / "src"
    src.mkdir()
    
    calc_py = src / "calc.py"
    calc_py.write_text(
        "def divide(a: int, b: int) -> float:\n"
        "    if b == 0:\n"
        "        return 0.0\n"
        "    return a / b\n",
        encoding="utf-8",
    )
    
    tests = repo / "tests"
    tests.mkdir()
    
    test_calc = tests / "test_calc.py"
    test_calc.write_text(
        "from src.calc import divide\n\n"
        "def test_divide():\n"
        "    assert divide(10, 2) == 5.0\n"
        "    assert divide(5, 0) == 0.0\n",
        encoding="utf-8",
    )
    
    pyproject = repo / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'sample'\nversion = '0.1.0'\n", encoding="utf-8")
    
    return repo


@pytest.fixture
def sample_contract() -> RequirementContract:
    """Standard valid RequirementContract for testing."""
    return RequirementContract(
        task_type="bug_fix",
        observed_behavior="ZeroDivisionError raised when b is zero in divide(a, b)",
        expected_behavior="divide(a, 0) returns 0.0 without raising ZeroDivisionError",
        inviolable_constraints=["Maintain float return type", "Do not alter signature"],
        acceptance_oracles=["test_divide_by_zero_returns_zero"],
    )
