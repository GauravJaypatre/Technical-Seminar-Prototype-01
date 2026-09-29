"""Tests verifying RepoSandbox isolation, diff validation, and byte-level repository immutability."""

import hashlib
from pathlib import Path
import pytest

from src.config import AERSConfig
from src.model.inference import FakeLLM
from src.pipeline import AERSPipeline
from src.schemas import FailureDiagnosis, PatchResult
from src.stubs.sandbox_stub import RepoSandboxStub, apply_unified_diff_to_text
from src.stubs.test_runner_stub import TestRunnerStub


def hash_directory_files(root: Path) -> dict[str, str]:
    """Computes SHA256 hashes for all files in a directory."""
    hashes = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.parts):
            hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def test_pipeline_leaves_source_repo_byte_identical(tmp_path: Path):
    """Verify source repository is 100% byte-identical after full pipeline execution."""
    # Create a small realistic target repo
    repo = tmp_path / "target_repo"
    src_dir = repo / "src"
    tests_dir = repo / "tests"
    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    calc_file = src_dir / "calc.py"
    calc_file.write_text("def divide(a, b):\n    return a / b\n", encoding="utf-8")

    test_file = tests_dir / "test_calc.py"
    test_file.write_text(
        "from src.calc import divide\ndef test_div(): assert divide(4, 2) == 2\n",
        encoding="utf-8",
    )

    hashes_before = hash_directory_files(repo)
    assert len(hashes_before) == 2

    # 1. Run pipeline in passing mode
    config = AERSConfig(dry_run=True)
    fake_engine = FakeLLM()
    fake_diff = (
        "--- a/src/calc.py\n"
        "+++ b/src/calc.py\n"
        "@@ -1,2 +1,4 @@\n"
        " def divide(a, b):\n"
        "+    if b == 0:\n"
        "+        return 0\n"
        "     return a / b\n"
    )
    fake_engine.set_response("stage6_patch", fake_diff)

    runner = TestRunnerStub(force_pass=True)
    pipeline = AERSPipeline(config=config, engine=fake_engine)
    pipeline.registry.test_runner = runner

    result = pipeline.run(issue="ZeroDivisionError in divide()", repo_path=repo)
    assert isinstance(result, PatchResult)

    hashes_after_pass = hash_directory_files(repo)
    assert hashes_before == hashes_after_pass, "Source repo was modified during successful pipeline run!"

    # 2. Run pipeline in failing mode (exhausting retries)
    failing_runner = TestRunnerStub(force_pass=False)
    pipeline.registry.test_runner = failing_runner

    fail_result = pipeline.run(issue="ZeroDivisionError in divide()", repo_path=repo)
    assert isinstance(fail_result, FailureDiagnosis)

    hashes_after_fail = hash_directory_files(repo)
    assert hashes_before == hashes_after_fail, "Source repo was modified during failed pipeline run!"


def test_apply_diff_malformed_hunk_counts():
    """Verify fuzzy context applier handles imperfect line count headers from 3B models."""
    original = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    # Claimed hunk says -1,10 +1,12 when there are only 5 lines
    diff = (
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "@@ -1,10 +1,12 @@\n"
        " line 1\n"
        "+line 1.5\n"
        " line 2\n"
        " line 3\n"
        " line 4\n"
        " line 5\n"
    )
    result = apply_unified_diff_to_text(original, diff)
    assert "line 1.5" in result
    assert result.startswith("line 1\nline 1.5\nline 2\n")


def test_apply_diff_wrong_context_lines_rejected():
    """Verify patches with hallucinated/mismatched context lines are strictly rejected."""
    original = "def add(a, b):\n    return a + b\n"
    # Context lines do not exist in original
    diff = (
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "@@ -1,3 +1,4 @@\n"
        " def multiply_numbers(x, y, z):\n"
        "+    # new comment\n"
        "     return x * y * z\n"
    )
    with pytest.raises(ValueError, match="Context mismatch"):
        apply_unified_diff_to_text(original, diff)


def test_sandbox_fuzzy_patch_rejected_if_no_match(tmp_path: Path):
    """Verify sandbox rejects patches where no context can be matched."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("print('hello')\n", encoding="utf-8")

    sandbox = RepoSandboxStub()
    sandbox.prepare(repo)

    bad_patch = (
        "--- a/sample.py\n"
        "+++ b/sample.py\n"
        "@@ -1,2 +1,3 @@\n"
        " completely nonexistent line 1\n"
        "+added line\n"
        " completely nonexistent line 2\n"
    )
    applied = sandbox.apply_patch(bad_patch)
    assert applied is False
    assert sandbox.fuzzy_applied is False
    sandbox.cleanup()


def test_sandbox_regenerates_canonical_diff(tmp_path: Path):
    """Verify sandbox regenerates canonical diff that applies cleanly via git apply --check."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    sandbox = RepoSandboxStub()
    sandbox.prepare(repo)

    patch = (
        "--- a/sample.py\n"
        "+++ b/sample.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def hello():\n"
        "+    # greeting\n"
        "     return 'world'\n"
    )
    applied = sandbox.apply_patch(patch)
    assert applied is True
    assert len(sandbox.canonical_diff) > 0
    assert "--- a/sample.py" in sandbox.canonical_diff
    assert "+++ b/sample.py" in sandbox.canonical_diff
    assert "+    # greeting" in sandbox.canonical_diff
    sandbox.cleanup()
