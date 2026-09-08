"""
Unit tests for the execution sandbox.
"""
import os
import pytest
from src.sandbox import run_in_sandbox, SandboxResult

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TASK_EASY_01 = os.path.join(BASE_DIR, "data", "mock_repos", "easy", "easy_01")

def test_sandbox_unpatched_fails():
    """Verify that easy_01 in its unpatched state fails pytest in sandbox."""
    res = run_in_sandbox(TASK_EASY_01)
    assert res.success is False
    assert res.exit_code != 0
    assert any("test_empty_value_retained" in t for t in res.failed_tests)

def test_sandbox_ground_truth_patch_passes():
    """Verify that easy_01 with ground truth patch passes pytest in sandbox."""
    import json
    with open(os.path.join(TASK_EASY_01, "metadata.json"), "r") as f:
        meta = json.load(f)
    patch = meta["ground_truth_patch"]

    res = run_in_sandbox(TASK_EASY_01, patch_str=patch)
    assert res.success is True
    assert res.exit_code == 0
    assert len(res.failed_tests) == 0
    assert len(res.passed_tests) > 0

def test_sandbox_malformed_patch():
    """Verify that a malformed patch fails gracefully without crashing."""
    res = run_in_sandbox(TASK_EASY_01, patch_str="MALFORMED GARBAGE PATCH")
    assert res.success is False
    assert res.patch_applied is False
    assert "Patch application failed" in res.stderr
