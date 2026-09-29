"""Stub implementation of RepoSandbox for patch isolation and rollback.

# TODO(member-4): Replace with transactional git worktree or container sandbox.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import subprocess

from src.interfaces import IRepoSandbox

logger = logging.getLogger(__name__)


def apply_unified_diff_to_text(original_text: str, diff_text: str) -> str:
    """Context-aware pure-Python unified diff applier with fuzzy line matching."""
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
            continue

        # Strategy 1: Find exact block match in new_lines
        match_idx = -1
        block_len = len(expected_old)

        # Search near hint_start first
        search_range = list(range(max(0, hint_start - 10), min(len(new_lines) - block_len + 1, hint_start + 10)))
        # Then search entire file
        search_range += [i for i in range(len(new_lines) - block_len + 1) if i not in search_range]

        for idx in search_range:
            if [l.strip() for l in new_lines[idx : idx + block_len]] == [l.strip() for l in expected_old]:
                match_idx = idx
                break

        # Strategy 2: If block not found, match by first context / deletion line
        if match_idx == -1 and expected_old:
            first_target = expected_old[0].strip()
            for idx in range(len(new_lines)):
                if new_lines[idx].strip() == first_target:
                    match_idx = idx
                    break

        # Strategy 3: Fallback to hint_start
        if match_idx == -1:
            match_idx = min(hint_start, len(new_lines))

        # Apply replacement at matched index
        del new_lines[match_idx : match_idx + block_len]
        for offset, added_line in enumerate(replacement):
            new_lines.insert(match_idx + offset, added_line)

    return "\n".join(new_lines) + ("\n" if original_text.endswith("\n") else "")


class RepoSandboxStub(IRepoSandbox):
    """Stub implementation of RepoSandbox supporting git and transactional in-memory backups."""

    def __init__(self) -> None:
        self.repo_path: Path | None = None
        self._backups: dict[Path, str] = {}
        self._created_files: list[Path] = []

    def prepare(self, repo: str | Path) -> None:
        # TODO(member-4): Create isolated worktree or shadow clone
        self.repo_path = Path(repo).resolve()
        self._backups.clear()
        self._created_files.clear()

        if self.repo_path.exists():
            for root, dirs, files in os.walk(self.repo_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", ".venv", "__pycache__")]
                for f in files:
                    p = Path(root) / f
                    try:
                        self._backups[p] = p.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        pass
        logger.info("RepoSandbox prepared for %s (%d files backed up)", self.repo_path, len(self._backups))

    def apply_patch(self, patch: str) -> bool:
        # TODO(member-4): Apply git diff cleanly to worktree
        if not self.repo_path or not self.repo_path.exists() or not patch.strip():
            return False

        # Attempt 1: Try native git apply
        try:
            proc = subprocess.run(
                ["git", "apply", "--ignore-whitespace", "-"],
                cwd=str(self.repo_path),
                input=patch,
                text=True,
                capture_output=True,
            )
            if proc.returncode == 0:
                logger.info("Successfully applied patch via git apply")
                return True
            else:
                logger.warning("git apply failed: %s. Attempting context-aware direct application.", proc.stderr.strip())
        except Exception as e:
            logger.warning("Error invoking git apply: %s", e)

        # Attempt 2: Direct context-aware file patching fallback
        try:
            target_match = re.search(r"^\+\+\+ [ab]/(.+)$", patch, re.MULTILINE)
            if not target_match:
                target_match = re.search(r"^\+\+\+ (.+)$", patch, re.MULTILINE)

            if target_match:
                rel_file = target_match.group(1).strip()
                target_file = self.repo_path / rel_file
                if target_file.exists():
                    orig_content = target_file.read_text(encoding="utf-8")
                    new_content = apply_unified_diff_to_text(orig_content, patch)
                    target_file.write_text(new_content, encoding="utf-8")
                    logger.info("Successfully applied patch directly to %s", target_file)
                    return True
        except Exception as e:
            logger.error("Direct fallback patch application failed: %s", e)

        return False

    def rollback(self) -> None:
        # TODO(member-4): Revert changes or reset worktree
        if not self.repo_path or not self.repo_path.exists():
            return

        # Restore from backup snapshot
        for path, content in self._backups.items():
            try:
                path.write_text(content, encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to restore %s during rollback: %s", path, e)

        for path in self._created_files:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

        try:
            subprocess.run(
                ["git", "checkout", "."],
                cwd=str(self.repo_path),
                capture_output=True,
            )
        except Exception:
            pass

        logger.info("Rolled back sandbox changes to clean state.")
