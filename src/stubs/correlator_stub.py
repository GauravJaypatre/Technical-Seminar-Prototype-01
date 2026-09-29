"""Stub implementation of TestLogCorrelator for AST/Traceback mapping.

# TODO(member-2/3): Replace with AST node-mapping correlator.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re

from src.interfaces import ITestLogCorrelator
from src.schemas import SuspiciousRange

logger = logging.getLogger(__name__)


class TestLogCorrelatorStub(ITestLogCorrelator):
    """Parses Python stack traces to locate referenced files and line numbers."""
    __test__ = False

    def correlate(self, traceback: str, repo: str | Path) -> list[SuspiciousRange]:
        # TODO(member-2/3): Implement full AST-level mapping
        repo_path = Path(repo).resolve()
        results: list[SuspiciousRange] = []

        if not traceback:
            return results

        # Regex to capture File "path/to/file.py", line 123
        pattern = re.compile(r'File\s+["\']([^"\']+)["\'],\s+line\s+(\d+)')
        for match in pattern.finditer(traceback):
            file_str, line_str = match.groups()
            line = int(line_str)
            p = Path(file_str)

            # Check if relative to repo
            rel_path: str | None = None
            if p.is_absolute():
                try:
                    rel_path = p.relative_to(repo_path).as_posix()
                except ValueError:
                    rel_path = None
            else:
                if (repo_path / p).exists():
                    rel_path = p.as_posix()

            if rel_path:
                results.append(
                    SuspiciousRange(
                        file=rel_path,
                        start=max(1, line - 5),
                        end=line + 5,
                        confidence=0.85,
                    )
                )

        logger.info("TestLogCorrelatorStub found %d suspicious ranges in traceback", len(results))
        return results
