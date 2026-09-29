"""Tests for schemas, RequirementContract, and EvidenceBaton projection."""

import pytest
from pydantic import ValidationError

from src.schemas import (
    BATON_HARD_CAP_TOKENS,
    INITIAL_BATON_MAX_TOKENS,
    EvidenceBaton,
    Hypothesis,
    LocalizationResult,
    PatchResult,
    RequirementContract,
    SuspiciousRange,
)


def test_requirement_contract_valid(sample_contract: RequirementContract):
    """Test valid RequirementContract has exact fields."""
    assert sample_contract.task_type == "bug_fix"
    assert "ZeroDivisionError" in sample_contract.observed_behavior
    assert len(sample_contract.inviolable_constraints) == 2
    assert len(sample_contract.acceptance_oracles) == 1


def test_requirement_contract_extra_fields_forbidden():
    """Verify extra fields are strictly forbidden."""
    with pytest.raises(ValidationError):
        RequirementContract(
            task_type="bug_fix",
            observed_behavior="broken",
            expected_behavior="fixed",
            inviolable_constraints=[],
            acceptance_oracles=[],
            unexpected_field="disallowed",  # type: ignore[call-arg]
        )


def test_requirement_contract_task_type_normalization():
    """Verify task_type normalizes gracefully."""
    c1 = RequirementContract(
        task_type="BUG_FIX",
        observed_behavior="x",
        expected_behavior="y",
    )
    assert c1.task_type == "bug_fix"

    c2 = RequirementContract(
        task_type="new_feature",
        observed_behavior="x",
        expected_behavior="y",
    )
    assert c2.task_type == "feature"


def test_initial_baton_within_500_tokens(sample_contract: RequirementContract):
    """Verify that initial baton B0 adheres to <= 500 token budget."""
    baton = EvidenceBaton(requirement=sample_contract)
    tokens = baton.token_count()
    assert tokens <= INITIAL_BATON_MAX_TOKENS
    assert tokens > 0


def test_baton_json_roundtrip(sample_contract: RequirementContract):
    """Verify serialization and deserialization preserves all fields."""
    b0 = EvidenceBaton(
        requirement=sample_contract,
        candidate_files=["src/calc.py"],
        suspicious_ranges=[SuspiciousRange(file="src/calc.py", start=1, end=5, confidence=0.9)],
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="Missing zero check",
                failure_mechanism="div by 0",
                falsification_test="test non-zero",
                status="PROPOSED",
                score=0.85,
            )
        ],
    )
    json_str = b0.to_json()
    b_loaded = EvidenceBaton.from_json(json_str)

    assert b_loaded.candidate_files == ["src/calc.py"]
    assert len(b_loaded.suspicious_ranges) == 1
    assert b_loaded.hypotheses[0].id == "H1"
    assert b_loaded.requirement.task_type == "bug_fix"


def test_baton_projection_enforces_caps(sample_contract: RequirementContract):
    """Verify project() caps files <= 5, hypotheses <= 3, and compresses traces."""
    # Create an oversized baton
    files = [f"src/file_{i}.py" for i in range(10)]
    ranges = [
        SuspiciousRange(file=f"src/file_{i}.py", start=i * 10, end=i * 10 + 5, confidence=i / 10.0)
        for i in range(10)
    ]
    hypotheses = [
        Hypothesis(
            id=f"H{i}",
            statement=f"Hypothesis statement {i}",
            failure_mechanism=f"Failure mechanism details {i}",
            falsification_test=f"Test condition {i}",
            score=0.5,
        )
        for i in range(7)
    ]
    large_trace = "Traceback (most recent call last):\n" + ("  File 'foo.py', line 12\n" * 50)

    baton = EvidenceBaton(
        requirement=sample_contract,
        candidate_files=files,
        suspicious_ranges=ranges,
        hypotheses=hypotheses,
        retry_traces=[large_trace],
    )

    projected = baton.project(max_tokens=BATON_HARD_CAP_TOKENS)

    assert len(projected.candidate_files) <= 5
    assert len(projected.suspicious_ranges) <= 5
    assert len(projected.hypotheses) <= 3
    assert len(projected.retry_traces) <= 1
    assert projected.token_count() <= BATON_HARD_CAP_TOKENS


def test_baton_append_traceback(sample_contract: RequirementContract):
    """Verify append_traceback retains only the latest error and enforces cap."""
    b = EvidenceBaton(requirement=sample_contract)
    b1 = b.append_traceback("First error traceback")
    assert len(b1.retry_traces) == 1
    assert "First error" in b1.retry_traces[0]

    b2 = b1.append_traceback("Second error traceback that is newer")
    assert len(b2.retry_traces) == 1
    assert "Second error" in b2.retry_traces[0]
    assert "First error" not in b2.retry_traces[0]


def test_baton_project_raises_when_unfit():
    """Verify ValueError is raised if requirement contract alone exceeds max_tokens."""
    huge_contract = RequirementContract(
        task_type="bug_fix",
        observed_behavior="x" * 10000,
        expected_behavior="y" * 10000,
        inviolable_constraints=["z" * 1000 for _ in range(20)],
        acceptance_oracles=[],
    )
    b = EvidenceBaton(requirement=huge_contract)
    with pytest.raises(ValueError, match="could not be projected"):
        b.project(max_tokens=200)
