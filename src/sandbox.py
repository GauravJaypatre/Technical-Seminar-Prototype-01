"""
Isolated execution sandbox for benchmarking code patches.
Features:
- Ephemeral execution in a temporary directory
- Resilient patch application (unified diff / direct file rewrite)
- Strict process execution timeout (default: 15.0 seconds)
- Pytest output parsing (passed/failed tests, tracebacks)
"""
import os
import re
import sys
import shutil
import tempfile
import subprocess
from typing import Optional, List, Dict, Any

class SandboxResult:
    def __init__(
        self,
        success: bool,
        exit_code: int,
        stdout: str,
        stderr: str,
        execution_time_sec: float,
        passed_tests: List[str] = None,
        failed_tests: List[str] = None,
        patch_applied: bool = True,
        patch_error: Optional[str] = None,
        used_fallback: bool = False
    ):
        self.success = success
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time_sec = execution_time_sec
        self.passed_tests = passed_tests or []
        self.failed_tests = failed_tests or []
        self.patch_applied = patch_applied
        self.patch_error = patch_error
        self.used_fallback = used_fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "execution_time_sec": round(self.execution_time_sec, 3),
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "patch_applied": self.patch_applied,
            "patch_error": self.patch_error,
            "used_fallback": self.used_fallback,
            "stdout_snippet": self.stdout[-1000:] if self.stdout else "",
            "stderr_snippet": self.stderr[-1000:] if self.stderr else ""
        }

def clean_diff_text(patch_text: str) -> str:
    """Extracts unified diff content from LLM markdown code blocks if present."""
    if not patch_text:
        return ""
    # Extract from markdown code fence: ```diff, ```patch, ```python, or bare ```
    match = re.search(r"```(?:diff|patch|[a-zA-Z0-9_-]+)?\s*\n(.*?)\n```", patch_text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if "--- " in extracted or "diff --git" in extracted:
            # Strip trailing pseudo-markers like '*** End of File'
            extracted = re.sub(r"(?:\r?\n)\*\*\*.*$", "", extracted, flags=re.MULTILINE).strip()
            return extracted
    # If not found in fence, look for diff headers in the raw text
    diff_start = re.search(r"(?:^|\n)(--- [ab]/[^\n]+|\+\+\+ [ab]/[^\n]+|diff --git [^\n]+)", patch_text)
    if diff_start:
        raw_diff = patch_text[diff_start.start():].strip()
        if raw_diff.endswith("```"):
            raw_diff = raw_diff[:-3].strip()
        raw_diff = re.sub(r"(?:\r?\n)\*\*\*.*$", "", raw_diff, flags=re.MULTILINE).strip()
        return raw_diff
    return patch_text.strip()

def apply_unified_diff(repo_dir: str, patch_str: str) -> tuple[bool, Optional[str], bool]:
    """
    Applies a unified diff patch to files in repo_dir using a Python-native patcher
    with fallback to git apply.
    Returns: (patch_applied: bool, error_message: Optional[str], used_fallback: bool)
    """
    cleaned_patch = clean_diff_text(patch_str)
    if not cleaned_patch:
        return False, "Empty patch content provided.", False

    # Try applying via git apply first if git is available and repo has git tracking
    patch_file = os.path.join(repo_dir, "_temp_eval.patch")
    with open(patch_file, "w", encoding="utf-8") as f:
        f.write(cleaned_patch + "\n")

    try:
        proc = subprocess.run(
            ["git", "apply", "--ignore-space-change", "--ignore-whitespace", "--recount", "_temp_eval.patch"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        if proc.returncode == 0:
            if os.path.exists(patch_file):
                os.remove(patch_file)
            return True, None, False
    except Exception:
        pass

    # Python-native unified diff applier for single/multi-file diffs
    try:
        applied_any = False
        used_fallback = False
        chunks = re.split(r"(?=^--- )", cleaned_patch, flags=re.MULTILINE)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or not chunk.startswith("--- "):
                continue

            lines = chunk.splitlines()
            target_file = None
            for line in lines[:4]:
                if line.startswith("+++ "):
                    raw_target = line[4:].strip()
                    # Strip leading a/ or b/
                    if raw_target.startswith("b/") or raw_target.startswith("a/"):
                        raw_target = raw_target[2:]
                    target_file = os.path.join(repo_dir, raw_target)
                    break

            if not target_file or not os.path.exists(target_file):
                # Try finding matching relative path
                possible = [os.path.join(repo_dir, l[4:].strip().lstrip("b/").lstrip("a/")) for l in lines[:4] if l.startswith("+++ ")]
                if possible and os.path.exists(possible[0]):
                    target_file = possible[0]
                else:
                    continue

            with open(target_file, "r", encoding="utf-8") as tf:
                original_content = tf.read()

            new_content, chunk_fallback = _apply_hunks_to_text(original_content, lines)
            if new_content is not None and new_content != original_content:
                with open(target_file, "w", encoding="utf-8") as tf:
                    tf.write(new_content)
                applied_any = True
                if chunk_fallback:
                    used_fallback = True

        if os.path.exists(patch_file):
            os.remove(patch_file)

        if applied_any:
            return True, None, used_fallback
        return False, "Failed to locate target file or apply diff hunks.", False
    except Exception as e:
        if os.path.exists(patch_file):
            os.remove(patch_file)
        return False, f"Exception while applying patch: {str(e)}", False

def _apply_hunks_to_text(original_text: str, patch_lines: List[str]) -> tuple[Optional[str], bool]:
    """Applies unified diff hunks to string content. Returns (new_content, used_fallback)."""
    orig_lines = original_text.splitlines()
    hunk_indices = [i for i, line in enumerate(patch_lines) if line.startswith("@@")]
    if not hunk_indices:
        return None, False

    modified_any = False
    used_fallback = False

    # Process hunks
    for idx_pos in range(len(hunk_indices)):
        start_idx = hunk_indices[idx_pos]
        end_idx = hunk_indices[idx_pos + 1] if idx_pos + 1 < len(hunk_indices) else len(patch_lines)
        hunk_header = patch_lines[start_idx]
        hunk_content = patch_lines[start_idx + 1:end_idx]

        # Extract line numbers @@ -start,len +start,len @@ if present
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", hunk_header)
        orig_start = int(m.group(1)) - 1 if m else 0
        if not m:
            used_fallback = True

        old_hunk_lines = []
        new_hunk_lines = []
        for l in hunk_content:
            if l.startswith("-"):
                old_hunk_lines.append(l[1:])
            elif l.startswith("+"):
                new_hunk_lines.append(l[1:])
            elif l.startswith(" "):
                old_hunk_lines.append(l[1:])
                new_hunk_lines.append(l[1:])
            elif not l.startswith("\\"):
                # Handle omitted diff context space gracefully
                old_hunk_lines.append(l)
                new_hunk_lines.append(l)

        # Match hunk position: try exact line, then search in vicinity
        match_idx = -1
        for search_offset in range(len(orig_lines)):
            for sign in [1, -1]:
                cand = orig_start + sign * search_offset
                if 0 <= cand <= len(orig_lines) - len(old_hunk_lines):
                    if [l.rstrip() for l in orig_lines[cand:cand + len(old_hunk_lines)]] == [l.rstrip() for l in old_hunk_lines]:
                        match_idx = cand
                        break
            if match_idx != -1:
                break

        if match_idx != -1:
            orig_lines[match_idx:match_idx + len(old_hunk_lines)] = new_hunk_lines
            modified_any = True
        else:
            # Fuzzy match: replace first occurrence of old lines if unique
            old_block = "\n".join(old_hunk_lines).strip()
            new_block = "\n".join(new_hunk_lines).strip()
            curr_text = "\n".join(orig_lines)
            if old_block and old_block in curr_text:
                curr_text = curr_text.replace(old_block, new_block, 1)
                orig_lines = curr_text.splitlines()
                modified_any = True
                used_fallback = True

    if not modified_any:
        return None, False

    res = "\n".join(orig_lines) + ("\n" if original_text.endswith("\n") else "")
    if res == original_text:
        return None, False
    return res, used_fallback

def parse_pytest_output(stdout: str, stderr: str) -> tuple[List[str], List[str]]:
    """Parses pytest terminal output to extract passed and failed test names."""
    passed = []
    failed = []
    full_output = stdout + "\n" + stderr
    for line in full_output.splitlines():
        line = line.strip()
        if " PASSED" in line:
            test_name = line.split()[0]
            passed.append(test_name)
        elif " FAILED" in line or " ERROR" in line:
            test_name = line.split()[0]
            failed.append(test_name)
    return passed, failed

def run_in_sandbox(
    task_repo_dir: str,
    patch_str: Optional[str] = None,
    timeout_sec: float = 15.0
) -> SandboxResult:
    """
    Copies the repository to an isolated temporary folder, applies the optional patch,
    and runs pytest within the sandbox.
    """
    import time
    start_time = time.time()

    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox_repo = os.path.join(temp_dir, "repo")
        shutil.copytree(task_repo_dir, sandbox_repo)

        # Initialize git in the sandbox replica so git apply works reliably
        try:
            subprocess.run(["git", "init"], cwd=sandbox_repo, capture_output=True, text=True, check=False)
            subprocess.run(["git", "config", "user.name", "sandbox"], cwd=sandbox_repo, capture_output=True, text=True, check=False)
            subprocess.run(["git", "config", "user.email", "sandbox@bench.local"], cwd=sandbox_repo, capture_output=True, text=True, check=False)
            subprocess.run(["git", "add", "-A"], cwd=sandbox_repo, capture_output=True, text=True, check=False)
            subprocess.run(["git", "commit", "-m", "initial", "--quiet"], cwd=sandbox_repo, capture_output=True, text=True, check=False)
        except Exception:
            pass

        used_fallback = False
        # Apply patch if provided
        if patch_str:
            patch_success, patch_err, used_fallback = apply_unified_diff(sandbox_repo, patch_str)
            if not patch_success:
                elapsed = time.time() - start_time
                return SandboxResult(
                    success=False,
                    exit_code=2,
                    stdout="",
                    stderr=f"Patch application failed: {patch_err}",
                    execution_time_sec=elapsed,
                    patch_applied=False,
                    patch_error=patch_err,
                    used_fallback=used_fallback
                )

        # Execute pytest
        env = os.environ.copy()
        # Ensure python can import from sandbox_repo
        env["PYTHONPATH"] = sandbox_repo + os.pathsep + env.get("PYTHONPATH", "")

        try:
            cmd = [sys.executable, "-m", "pytest", "-v", "--no-header", "--no-summary", "tests"]
            proc = subprocess.run(
                cmd,
                cwd=sandbox_repo,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env
            )
            elapsed = time.time() - start_time
            passed, failed = parse_pytest_output(proc.stdout, proc.stderr)
            success = (proc.returncode == 0) and (len(failed) == 0)

            return SandboxResult(
                success=success,
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                execution_time_sec=elapsed,
                passed_tests=passed,
                failed_tests=failed,
                patch_applied=True,
                used_fallback=used_fallback
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return SandboxResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Pytest execution timed out after {timeout_sec} seconds.",
                execution_time_sec=elapsed,
                patch_applied=True,
                used_fallback=used_fallback
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return SandboxResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=f"Sandbox execution error: {str(e)}",
                execution_time_sec=elapsed,
                patch_applied=True,
                used_fallback=used_fallback
            )
