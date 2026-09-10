"""
Unit tests enforcing permanent offline simulation fallback neutralization.
Guarantees that the codebase cannot silently fabricate benchmark figures or inject ground truth patches.
Any attempt to invoke offline simulation paths must raise a hard-failing RuntimeError.
"""
import os
import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.sandbox import SandboxResult


def test_monolithic_baseline_mock_frontier_raises_runtime_error():
    """Verify call_mock_frontier in src/monolithic_baseline.py raises RuntimeError."""
    from src.monolithic_baseline import call_mock_frontier
    with pytest.raises(RuntimeError, match="neutralized"):
        call_mock_frontier("dummy_dir", seed=42)


def test_monolithic_baseline_offline_run_raises_runtime_error():
    """Verify run_monolithic_baseline with live=False raises RuntimeError."""
    from src.monolithic_baseline import run_monolithic_baseline
    with pytest.raises(RuntimeError, match="neutralized"):
        run_monolithic_baseline(live=False)


def test_code_analyzer_offline_raises_runtime_error(monkeypatch):
    """Verify CodeAnalyzerAgent._call_llm raises RuntimeError when Ollama is offline."""
    import src.code_analyzer as ca
    monkeypatch.setattr(ca, "_is_ollama_online", lambda: False)
    
    agent = ca.CodeAnalyzerAgent()
    with pytest.raises(RuntimeError, match="neutralized"):
        agent._call_llm(prompt="test prompt", meta={}, iteration=1, qa_feedback=None, seed=42)


def test_router_offline_raises_runtime_error(monkeypatch):
    """Verify RouterAgent._call_llm raises RuntimeError when Ollama is offline."""
    import src.router as rt
    monkeypatch.setattr(rt, "_is_ollama_online", lambda: False)
    
    router = rt.RouterAgent()
    with pytest.raises(RuntimeError, match="neutralized"):
        router._call_llm(prompt="test prompt", meta={})


def test_qa_verifier_offline_raises_runtime_error(monkeypatch):
    """Verify QAVerifierAgent._generate_feedback raises RuntimeError when Ollama is offline."""
    import src.qa_verifier as qv
    monkeypatch.setattr(qv, "_is_ollama_online", lambda: False)
    
    verifier = qv.QAVerifierAgent()
    dummy_res = SandboxResult(
        success=False,
        exit_code=1,
        stdout="AssertionError: test failed",
        stderr="",
        execution_time_sec=0.1,
        patch_applied=True,
        failed_tests=["test_item"],
        passed_tests=[]
    )
    with pytest.raises(RuntimeError, match="neutralized"):
        verifier._generate_feedback(dummy_res)


def test_swarm_runner_offline_raises_runtime_error():
    """Verify run_swarm_baseline with live=False raises RuntimeError."""
    from src.swarm.swarm_runner import run_swarm_baseline
    with pytest.raises(RuntimeError, match="neutralized"):
        run_swarm_baseline(live=False)


def test_src_repo_wide_fabrication_signatures_neutralized():
    """
    Assert that none of the synthetic fabrication signatures exist anywhere in src/.
    Scans every Python file under src/ recursively.
    """
    signatures = [
        "base_probs",
        "success_rates",
        "ground_truth_patch",
        "rng.random() <",
        "random.Random(seed",
    ]
    src_dir = BASE_DIR / "src"
    py_files = list(src_dir.rglob("*.py"))
    assert len(py_files) > 0, "No python source files found in src/"

    violations = []
    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        for sig in signatures:
            if sig in content:
                rel_path = py_file.relative_to(BASE_DIR).as_posix()
                violations.append(f"{rel_path} contains forbidden fabrication signature '{sig}'")

    assert not violations, "\n".join(violations)


def test_historical_runs_telemetry_live():
    """
    Assert that historical benchmark run CSVs used for headline claims contain
    explicit live-execution evidence (is_live == True, provider == ollama_local/groq,
    realistic wall clock latency >= 2.0s).
    """
    import pandas as pd

    v1_csv = BASE_DIR / "results" / "prototype_run_swarm_v1_header_only_qa" / "benchmark_prototype_swarm_20260909_175738.csv"
    v2_csv = BASE_DIR / "results" / "prototype_run_swarm_v2_content_validated_qa" / "benchmark_prototype_swarm_20260909_204053.csv"
    groq_csv = BASE_DIR / "results" / "prototype_run_groq" / "benchmark_prototype_monolithic_groq_20260909_164450.csv"

    assert v1_csv.exists(), "v1 swarm run CSV missing"
    assert v2_csv.exists(), "v2 swarm run CSV missing"
    assert groq_csv.exists(), "groq run CSV missing"

    df_v1 = pd.read_csv(v1_csv)
    df_v2 = pd.read_csv(v2_csv)
    df_groq = pd.read_csv(groq_csv)

    assert len(df_v1) == 90
    assert len(df_v2) == 90
    assert len(df_groq) == 90

    # Swarm v1 live telemetry
    assert (df_v1["is_live"] == True).all(), "v1 contains non-live records"
    assert (df_v1["provider"] == "ollama_local").all()
    assert df_v1["cumulative_wall_clock_sec"].min() >= 20.0

    # Swarm v2 live telemetry
    assert (df_v2["is_live"] == True).all(), "v2 contains non-live records"
    assert (df_v2["provider"] == "ollama_local").all()
    assert df_v2["cumulative_wall_clock_sec"].min() >= 20.0

    # Groq live telemetry
    assert (df_groq["provider"] == "groq").all()
    assert df_groq["cumulative_wall_clock_sec"].min() >= 2.0


def test_commit_message_headline_claims_accuracy():
    """Verify that HEAD commit message does not contain typo/inaccurate figures (e.g. 84.44%)."""
    import subprocess
    res = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=True
    )
    commit_msg = res.stdout
    assert "84.44%" not in commit_msg, "Commit message contains typo '84.44%' (expected 85.56%)"


