"""
Comprehensive verification test:
Validates that ALL 18 mock repositories:
1. Fail their targeted tests when unpatched.
2. Pass completely when their ground-truth patch is applied.
"""
import os
import json
import pytest
from src.sandbox import run_in_sandbox

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "task_manifest.json")

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    MANIFEST = json.load(f)

ALL_TASKS = MANIFEST["all_tasks"]

@pytest.mark.parametrize("task_info", ALL_TASKS, ids=[t["task_id"] for t in ALL_TASKS])
def test_mock_repo_unpatched_fails(task_info):
    """Verify that the mock repo fails tests in its initial unpatched state."""
    repo_dir = os.path.join(BASE_DIR, task_info["directory"])
    res = run_in_sandbox(repo_dir)
    assert res.success is False, f"Task {task_info['task_id']} unexpectedly passed while unpatched!"
    assert len(res.failed_tests) > 0, f"Task {task_info['task_id']} had no failing tests recorded!"

@pytest.mark.parametrize("task_info", ALL_TASKS, ids=[t["task_id"] for t in ALL_TASKS])
def test_mock_repo_ground_truth_patch_passes(task_info):
    """Verify that applying the ground-truth patch resolves all failing tests."""
    repo_dir = os.path.join(BASE_DIR, task_info["directory"])
    meta_path = os.path.join(repo_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    patch = meta["ground_truth_patch"]
    res = run_in_sandbox(repo_dir, patch_str=patch)
    assert res.patch_applied is True, f"Patch failed to apply for {task_info['task_id']}: {res.patch_error}"
    assert res.success is True, f"Task {task_info['task_id']} failed with patch! Error: {res.stderr}\nStdout: {res.stdout}"
    assert len(res.failed_tests) == 0
    assert len(res.passed_tests) > 0
