"""Tests for specialized agents (Stages 1, 3, 4, 6)."""

from pathlib import Path
import pytest

from src.agents.hypothesis import HypothesisAgent, HypothesisBatch
from src.agents.localizer import LocalizerAgent
from src.agents.patch_generator import PatchGeneratorAgent, extract_unified_diff, is_valid_diff_format
from src.agents.profiler import ProfilerAgent
from src.interfaces import RepoProfile
from src.model.inference import FakeLLM
from src.schemas import (
    EvidenceBaton,
    Hypothesis,
    LocalizationResult,
    RequirementContract,
    SuspiciousRange,
)
from src.stubs.correlator_stub import TestLogCorrelatorStub


def test_profiler_agent_fallback(fake_engine: FakeLLM):
    """Verify profiler agent falls back gracefully on model failure."""
    agent = ProfilerAgent(engine=fake_engine)
    fallback = agent.deterministic_fallback(
        issue="AttributeError in payment processor when card is expired",
        profile=RepoProfile(repo_path=".", language="python"),
    )
    assert fallback.task_type == "bug_fix"
    assert "AttributeError" in fallback.observed_behavior
    assert len(fallback.inviolable_constraints) > 0


def test_profiler_initial_baton_cap(fake_engine: FakeLLM, sample_contract: RequirementContract):
    """Verify initial baton B0 produced by profiler is <= 500 tokens."""
    agent = ProfilerAgent(engine=fake_engine)
    b0 = agent.create_initial_baton(sample_contract)
    assert b0.token_count() <= 500
    assert b0.requirement.task_type == "bug_fix"


def test_localizer_hallucination_filtering(fake_engine: FakeLLM, sample_repo: Path, sample_contract: RequirementContract):
    """Verify localizer agent drops hallucinated files that don't exist in the repo."""
    # Setup FakeLLM to return a mix of real and fake files
    fake_result = LocalizationResult(
        candidate_files=[
            "src/calc.py",            # Real file in sample_repo
            "non_existent/fake.py",   # Hallucinated file
            "another_hallucination.py",
        ],
        suspicious_ranges=[
            SuspiciousRange(file="src/calc.py", start=2, end=4, confidence=0.9),
            SuspiciousRange(file="non_existent/fake.py", start=10, end=20, confidence=0.8),
        ],
    )
    fake_engine.set_response("stage3_localizer", fake_result)

    agent = LocalizerAgent(engine=fake_engine)
    baton = EvidenceBaton(requirement=sample_contract)
    result = agent.run(baton=baton, context="", repo_path=sample_repo)

    # Hallucinated files must be dropped
    assert "src/calc.py" in result.candidate_files
    assert "non_existent/fake.py" not in result.candidate_files
    assert len(result.candidate_files) == 1
    assert len(result.suspicious_ranges) == 1
    assert result.suspicious_ranges[0].file == "src/calc.py"


def test_localizer_disambiguation_via_correlator(fake_engine: FakeLLM, sample_repo: Path, sample_contract: RequirementContract):
    """Verify correlator is invoked when localizer candidate list is empty and traceback exists."""
    fake_engine.set_response("stage3_localizer", LocalizationResult(candidate_files=[], suspicious_ranges=[]))

    correlator = TestLogCorrelatorStub()
    agent = LocalizerAgent(engine=fake_engine, correlator=correlator)

    baton = EvidenceBaton(
        requirement=sample_contract,
        retry_traces=[f'File "{sample_repo / "src" / "calc.py"}", line 3, in divide'],
    )
    result = agent.run(baton=baton, context="", repo_path=sample_repo)

    assert "src/calc.py" in result.candidate_files
    assert len(result.suspicious_ranges) > 0


def test_hypothesis_agent_max_3(fake_engine: FakeLLM, sample_contract: RequirementContract):
    """Verify hypothesis agent enforces <= 3 hypotheses with mechanisms."""
    batch = HypothesisBatch(
        hypotheses=[
            Hypothesis(
                id=f"H{i + 1}",
                statement=f"Hypothesis {i}",
                failure_mechanism=f"Mechanism {i}",
                falsification_test=f"Test {i}",
                score=0.9 - i * 0.1,
            )
            for i in range(6)
        ]
    )
    fake_engine.set_response("stage4_hypothesis", batch)

    agent = HypothesisAgent(engine=fake_engine)
    baton = EvidenceBaton(requirement=sample_contract)
    hyps = agent.run(baton=baton, context="")

    assert len(hyps) <= 3
    assert hyps[0].id == "H1"
    assert hyps[0].failure_mechanism == "Mechanism 0"
    assert hyps[0].falsification_test == "Test 0"


def test_patch_generator_unified_diff_validation():
    """Verify extraction and validation of unified diffs."""
    raw_response = (
        "Here is the fix:\n"
        "```diff\n"
        "--- a/src/calc.py\n"
        "+++ b/src/calc.py\n"
        "@@ -2,2 +2,4 @@\n"
        "+    if b == 0:\n"
        "+        return 0.0\n"
        "     return a / b\n"
        "```\n"
        "This resolves the division error."
    )
    diff = extract_unified_diff(raw_response)
    assert diff.startswith("--- a/src/calc.py")
    assert is_valid_diff_format(diff) is True


def test_patch_generator_retry_traceback_prompt(fake_engine: FakeLLM, sample_contract: RequirementContract):
    """Verify patch generator includes retry traceback in prompt when retrying."""
    agent = PatchGeneratorAgent(engine=fake_engine)
    baton = EvidenceBaton(
        requirement=sample_contract,
        retry_traces=["ZeroDivisionError: division by zero on line 42"],
    )

    agent.run(baton=baton, context="def divide(): ...", attempt=2)

    # Check that the call recorded in fake_engine contains the error
    last_call = fake_engine.calls[-1]
    assert "PREVIOUS ATTEMPT FAILED WITH ERROR" in last_call["user"]
    assert "ZeroDivisionError" in last_call["user"]


def test_localizer_candidate_files_capped_at_five(fake_engine: FakeLLM, tmp_path: Path, sample_contract: RequirementContract):
    """Verify localizer agent enforces hard cap of k <= 5 candidate files."""
    repo = tmp_path / "multi_file_repo"
    repo.mkdir()
    # Create 8 real files in repo
    real_files = []
    for i in range(8):
        f = repo / f"file_{i}.py"
        f.write_text(f"# file {i}\n", encoding="utf-8")
        real_files.append(f"file_{i}.py")

    fake_result = LocalizationResult(
        candidate_files=real_files,
        suspicious_ranges=[SuspiciousRange(file=rf, start=1, end=1, confidence=0.8) for rf in real_files],
    )
    fake_engine.set_response("stage3_localizer", fake_result)

    agent = LocalizerAgent(engine=fake_engine)
    baton = EvidenceBaton(requirement=sample_contract)
    result = agent.run(baton=baton, context="", repo_path=repo)

    assert len(result.candidate_files) <= 5
    assert len(result.candidate_files) == 5
    assert len(result.suspicious_ranges) <= 5
