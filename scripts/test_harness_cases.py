"""
Comprehensive harness verification script.
Tests patch parsing, CRLF/LF handling, multi-file patches, and unapplicable patch detection.
"""
import re
import os
import sys
import json
import difflib
import tempfile
import shutil
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.sandbox import clean_diff_text, apply_unified_diff, _apply_hunks_to_text, run_in_sandbox

def test_markdown_extraction():
    sample = "Commentary before\n```diff\n--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,1 +1,1 @@\n-old\n+new\n```\nCommentary after"
    assert clean_diff_text(sample) == "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,1 +1,1 @@\n-old\n+new"
    print("[PASS] Markdown code fence extraction: PASS")

def test_unapplicable_patch_detection():
    """
    CRITICAL AUDIT CHECK:
    Does apply_unified_diff correctly return (False, error) when given an unapplicable patch,
    or does it falsely return (True, None) while leaving files unchanged?
    """
    task_dir = os.path.join(BASE_DIR, "data", "mock_repos", "easy", "easy_01")
    with tempfile.TemporaryDirectory() as td:
        repo = os.path.join(td, "repo")
        shutil.copytree(task_dir, repo)
        
        # Completely bogus patch that cannot match
        bogus_patch = "--- a/src/query_parser.py\n+++ b/src/query_parser.py\n@@ -1,3 +1,3 @@\n-# NonExistentLine12345\n+# ReplacementLine\n"
        ok, err = apply_unified_diff(repo, bogus_patch)
        
        with open(os.path.join(repo, "src", "query_parser.py"), "r", encoding="utf-8") as f:
            content = f.read()
            
        modified = "# ReplacementLine" in content
        print("\n--- Unapplicable Patch Audit ---")
        print(f"  apply_unified_diff returned: ok={ok}, err={err}")
        print(f"  Was file actually modified?: {modified}")
        if ok is True and not modified:
            print("  [CRITICAL HARNESS BUG CONFIRMED]: apply_unified_diff falsely returns True even though the file was NOT modified!")
        else:
            print("  [PASS] Correctly rejected unapplicable patch.")

def test_crlf_handling():
    """Tests if CRLF line endings break diff application."""
    task_dir = os.path.join(BASE_DIR, "data", "mock_repos", "easy", "easy_01")
    with open(os.path.join(task_dir, "metadata.json")) as f:
        meta = json.load(f)
    patch_lf = meta["ground_truth_patch"]
    patch_crlf = patch_lf.replace("\n", "\r\n")

    res_lf = run_in_sandbox(task_dir, patch_str=patch_lf)
    res_crlf = run_in_sandbox(task_dir, patch_str=patch_crlf)
    print("\n--- Line Ending (CRLF vs LF) Audit ---")
    print(f"  LF Patch Success:   {res_lf.success}")
    print(f"  CRLF Patch Success: {res_crlf.success}")

def test_context_space_omission():
    """Tests what happens if Claude outputs context lines without leading spaces."""
    task_dir = os.path.join(BASE_DIR, "data", "mock_repos", "easy", "easy_03")
    # Missing leading space on context lines
    bad_context_patch = (
        "--- a/src/flag_validator.py\n"
        "+++ b/src/flag_validator.py\n"
        "@@ -3,4 +3,6 @@\n"
        "def validate_flag(flag: Optional[str], allowed_flags: Set[str]) -> bool:\n"
        "+    if flag is None:\n"
        "+        return False\n"
        "    normalized = flag.lower()\n"
    )
    res = run_in_sandbox(task_dir, patch_str=bad_context_patch)
    print("\n--- Context Space Omission Audit ---")
    print(f"  Missing context space patch applied: {res.patch_applied}")
    print(f"  Missing context space patch success: {res.success}")
    print(f"  Error: {res.stderr}")

if __name__ == "__main__":
    test_markdown_extraction()
    test_unapplicable_patch_detection()
    test_crlf_handling()
    test_context_space_omission()
