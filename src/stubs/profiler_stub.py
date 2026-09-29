"""Deterministic scan of the repository structure.

# TODO(member-2): Replace with full AST/heuristic repository profiler.
"""

from __future__ import annotations

import os
from pathlib import Path
import logging

from src.interfaces import IRepositoryProfiler, RepoProfile

logger = logging.getLogger(__name__)


class RepositoryProfilerStub(IRepositoryProfiler):
    """Stub implementation of RepositoryProfiler (Stage 1a)."""

    def profile(self, repo: str | Path) -> RepoProfile:
        # TODO(member-2): Add deep static analysis of package managers and frameworks
        repo_path = Path(repo).resolve()
        if not repo_path.exists():
            return RepoProfile(repo_path=str(repo_path))

        language = "python"
        build_system = "pip"
        test_framework = "pytest"

        # Basic heuristic scan
        files: list[str] = []
        for root, dirs, filenames in os.walk(repo_path):
            # Skip hidden / venv directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", ".venv", "__pycache__")]
            for f in filenames:
                rel = os.path.relpath(os.path.join(root, f), repo_path)
                files.append(rel.replace("\\", "/"))

        # Detect markers
        if any("pyproject.toml" in f for f in files):
            build_system = "pyproject.toml"
        elif any("setup.py" in f for f in files):
            build_system = "setup.py"
        elif any("package.json" in f for f in files):
            language = "javascript/typescript"
            build_system = "npm"
            test_framework = "jest"
        elif any("Cargo.toml" in f for f in files):
            language = "rust"
            build_system = "cargo"
            test_framework = "cargo test"

        logger.info("RepositoryProfilerStub scanned %s (%d files, %s/%s)", repo_path, len(files), language, build_system)
        return RepoProfile(
            repo_path=str(repo_path),
            language=language,
            build_system=build_system,
            test_framework=test_framework,
            file_tree=files[:100],  # Bound to top 100 paths
        )
