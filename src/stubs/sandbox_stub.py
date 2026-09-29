"""Stub implementation of RepoSandbox for patch isolation and rollback.

Provides isolated temporary sandbox copies of repositories, preventing any
modifications to the source repository. Supports native git apply with
context-aware fallback and canonical diff regeneration.
"""

from __future__ import annotations

import difflib
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from src.interfaces import IRepoSandbox

logger = logging.getLogger(__name__)


def apply_unified_diff_to_text(original_text: str, diff_text: str) -> str:
    """Context-aware pure-Python unified diff applier with fuzzy line matching.
    
    Raises ValueError if context lines cannot be matched within the target file.
    """
    original_lines = original_text.splitlines()
    hunk_regex = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    diff_lines = diff_text.splitlines()
    hunks: list[list[str]] = []
    current_hunk: list[str] = []

    for line in diff_lines:
        if line.startswith("@@ "):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = [line]
        elif current_hunk:
            current_hunk.append(line)
    if current_hunk:
        hunks.append(current_hunk)

    if not hunks:
        raise ValueError("No valid hunks found in unified diff.")

    new_lines = list(original_lines)

    for hunk in hunks:
        header = hunk[0]
        match = hunk_regex.match(header)
        hint_start = int(match.group(1)) - 1 if match else 0

        # Extract expected old lines (context + deletions)
        expected_old = [line[1:] for line in hunk[1:] if line.startswith(" ") or line.startswith("-")]
        # Extract new replacement lines (context + additions)
        replacement = [line[1:] for line in hunk[1:] if line.startswith(" ") or line.startswith("+")]

        if not expected_old:
            # Additions only without any context: verify hint_start
            idx = max(0, min(hint_start, len(new_lines)))
            for offset, added_line in enumerate(replacement):
                new_lines.insert(idx + offset, added_line)
            continue

        match_idx = -1
        block_len = len(expected_old)

        # Strategy 1: Find exact block match near hint_start first
        search_range = list(range(max(0, hint_start - 15), min(len(new_lines) - block_len + 1, hint_start + 15)))
        search_range += [i for i in range(len(new_lines) - block_len + 1) if i not in search_range]

        for idx in search_range:
            if [l.strip() for l in new_lines[idx : idx + block_len]] == [l.strip() for l in expected_old]:
                match_idx = idx
                break

        # Strategy 2: Match by first non-empty context/deletion line
        if match_idx == -1:
            first_target = expected_old[0].strip()
            for idx in range(len(new_lines) - block_len + 1):
                if new_lines[idx].strip() == first_target:
                    # Verify at least 60% of context lines match
                    matches = sum(
                        1 for o_line, e_line in zip(new_lines[idx : idx + block_len], expected_old)
                        if o_line.strip() == e_line.strip()
                    )
                    if matches / block_len >= 0.6:
                        match_idx = idx
                        break

        # If context could not be located anywhere in the file, reject patch
        if match_idx == -1:
            raise ValueError(
                f"Context mismatch: hunk context could not be located in target file.\n"
                f"Expected context:\n{chr(10).join(expected_old[:5])}"
            )

        # Apply replacement at matched index
        del new_lines[match_idx : match_idx + block_len]
        for offset, added_line in enumerate(replacement):
            new_lines.insert(match_idx + offset, added_line)

    return "\n".join(new_lines) + ("\n" if original_text.endswith("\n") else "")


class RepoSandboxStub(IRepoSandbox):
    """Isolated temporary sandbox copy of the repository.
    
    Guarantees the source repository is never modified during pipeline runs.
    Uses native git apply --check as primary validity check, falls back to
    fuzzy context application, and regenerates canonical unified diffs.
    """

    def __init__(self) -> None:
        self.source_repo: Path | None = None
        self.sandbox_path: Path | None = None
        self.fuzzy_applied: bool = False
        self.canonical_diff: str = ""

    def prepare(self, repo: str | Path) -> None:
        """Creates an isolated temporary copy of the target repository."""
        self.source_repo = Path(repo).resolve()
        if not self.source_repo.exists():
            raise FileNotFoundError(f"Target repository not found: {self.source_repo}")

        # Clean previous sandbox if any
        self.cleanup()

        # Create temporary sandbox directory
        temp_dir = Path(tempfile.mkdtemp(prefix="aers_sandbox_")).resolve()
        self.sandbox_path = temp_dir
        self.fuzzy_applied = False
        self.canonical_diff = ""

        # Copy repository contents ignoring version control and cache artifacts
        def ignore_patterns(path: str, names: list[str]) -> set[str]:
            ignored = set()
            for n in names:
                if n in (".git", "__pycache__", ".pytest_cache", ".venv", "venv", ".mypy_cache"):
                    ignored.add(n)
            return ignored

        shutil.copytree(self.source_repo, self.sandbox_path, dirs_exist_ok=True, ignore=ignore_patterns)

        # Initialize pristine git baseline in sandbox
        try:
            subprocess.run(["git", "init", "-b", "main"], cwd=str(self.sandbox_path), capture_output=True, check=True)
            subprocess.run(["git", "config", "user.email", "aers@local.test"], cwd=str(self.sandbox_path), capture_output=True)
            subprocess.run(["git", "config", "user.name", "AERS Sandbox"], cwd=str(self.sandbox_path), capture_output=True)
            subprocess.run(["git", "add", "."], cwd=str(self.sandbox_path), capture_output=True)
            subprocess.run(["git", "commit", "-m", "pristine baseline", "--allow-empty"], cwd=str(self.sandbox_path), capture_output=True)
        except Exception as e:
            logger.warning("Could not initialize git baseline in sandbox: %s", e)

        logger.info("RepoSandbox prepared at isolated path: %s (source: %s)", self.sandbox_path, self.source_repo)

    def apply_patch(self, patch: str) -> bool:
        """Applies a patch to the sandbox copy.
        
        Primary check: native `git apply --check`.
        Fallback: fuzzy context applier (logged with fuzzy_applied=True).
        Regenerates canonical diff on success.
        """
        if not self.sandbox_path or not self.sandbox_path.exists() or not patch.strip():
            return False

        self.fuzzy_applied = False
        self.canonical_diff = ""

        # Primary check: native git apply --check
        try:
            proc_check = subprocess.run(
                ["git", "apply", "--check", "--whitespace=nowarn", "-"],
                cwd=str(self.sandbox_path),
                input=patch,
                text=True,
                capture_output=True,
            )
            if proc_check.returncode == 0:
                proc_apply = subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", "-"],
                    cwd=str(self.sandbox_path),
                    input=patch,
                    text=True,
                    capture_output=True,
                )
                if proc_apply.returncode == 0:
                    self.fuzzy_applied = False
                    logger.info("Patch applied cleanly via native git apply")
                    self._regenerate_canonical_diff()
                    return True
                else:
                    logger.warning("git apply failed after passing check: %s", proc_apply.stderr)
            else:
                logger.warning(
                    "git apply --check failed (%s). Attempting fallback fuzzy context application.",
                    proc_check.stderr.strip(),
                )
        except Exception as e:
            logger.warning("Error running git apply: %s", e)

        # Fallback: Context-aware fuzzy patch applier
        try:
            target_match = re.search(r"^\+\+\+ [ab]/(.+)$", patch, re.MULTILINE)
            if not target_match:
                target_match = re.search(r"^\+\+\+ (.+)$", patch, re.MULTILINE)

            if target_match:
                rel_file = target_match.group(1).strip()
                target_file = self.sandbox_path / rel_file
                if target_file.exists():
                    orig_content = target_file.read_text(encoding="utf-8")
                    new_content = apply_unified_diff_to_text(orig_content, patch)
                    target_file.write_text(new_content, encoding="utf-8")
                    self.fuzzy_applied = True
                    logger.info('Patch applied via fuzzy context applier ("fuzzy_applied": true) to %s', rel_file)
                    self._regenerate_canonical_diff()
                    return True
                else:
                    logger.warning("Target file %s does not exist in sandbox.", rel_file)
        except Exception as e:
            logger.warning("Fuzzy fallback patch application rejected: %s", e)

        return False

    def _regenerate_canonical_diff(self) -> None:
        """Regenerates canonical unified diff from sandbox changes via git diff."""
        if not self.sandbox_path:
            return
        try:
            diff_proc = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(self.sandbox_path),
                text=True,
                capture_output=True,
            )
            raw_diff = diff_proc.stdout.strip()
            if raw_diff:
                self.canonical_diff = raw_diff
                logger.debug("Regenerated canonical diff (%d chars)", len(self.canonical_diff))
        except Exception as e:
            logger.warning("Could not regenerate canonical diff: %s", e)

    def rollback(self) -> None:
        """Resets sandbox changes back to pristine baseline commit."""
        if not self.sandbox_path or not self.sandbox_path.exists():
            return
        try:
            subprocess.run(["git", "checkout", "."], cwd=str(self.sandbox_path), capture_output=True)
            subprocess.run(["git", "clean", "-fd"], cwd=str(self.sandbox_path), capture_output=True)
        except Exception as e:
            logger.warning("Rollback error: %s", e)
        self.fuzzy_applied = False
        self.canonical_diff = ""
        logger.info("Rolled back sandbox changes to pristine baseline.")

    def cleanup(self) -> None:
        """Removes the temporary sandbox directory from disk."""
        if self.sandbox_path and self.sandbox_path.exists():
            try:
                shutil.rmtree(self.sandbox_path, ignore_errors=True)
                logger.debug("Deleted temporary sandbox at %s", self.sandbox_path)
            except Exception as e:
                logger.warning("Failed to clean up sandbox dir %s: %s", self.sandbox_path, e)
            self.sandbox_path = None
