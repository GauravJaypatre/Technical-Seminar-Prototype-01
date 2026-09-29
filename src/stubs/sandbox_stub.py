"""Stub implementation of RepoSandbox for patch isolation and rollback.

# TODO(member-4): Replace with transactional git worktree or container sandbox.
"""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess

from src.interfaces import IRepoSandbox

logger = logging.getLogger(__name__)


class RepoSandboxStub(IRepoSandbox):
    """Stub implementation of RepoSandbox."""

    def __init__(self) -> None:
        self.repo_path: Path | None = None
        self._modified_files: list[Path] = []

    def prepare(self, repo: str | Path) -> None:
        # TODO(member-4): Create isolated worktree or shadow clone
        self.repo_path = Path(repo).resolve()
        logger.info("RepoSandbox prepared for %s", self.repo_path)

    def apply_patch(self, patch: str) -> bool:
        # TODO(member-4): Apply git diff cleanly to worktree
        if not self.repo_path or not self.repo_path.exists():
            return False

        if not patch.strip():
            return False

        try:
            # Check if git is available
            proc = subprocess.run(
                ["git", "apply", "--check", "-"],
                cwd=str(self.repo_path),
                input=patch,
                text=True,
                capture_output=True,
            )
            if proc.returncode == 0:
                subprocess.run(
                    ["git", "apply", "-"],
                    cwd=str(self.repo_path),
                    input=patch,
                    text=True,
                    capture_output=True,
                )
                logger.info("Successfully applied patch via git apply")
                return True
            else:
                logger.warning("git apply --check failed: %s", proc.stderr)
                return False
        except Exception as e:
            logger.warning("Error running git apply in sandbox: %s", e)
            return False

    def rollback(self) -> None:
        # TODO(member-4): Revert changes or reset worktree
        if not self.repo_path or not self.repo_path.exists():
            return

        try:
            subprocess.run(
                ["git", "checkout", "."],
                cwd=str(self.repo_path),
                capture_output=True,
            )
            logger.info("Rolled back sandbox changes via git checkout .")
        except Exception as e:
            logger.warning("Rollback error: %s", e)
