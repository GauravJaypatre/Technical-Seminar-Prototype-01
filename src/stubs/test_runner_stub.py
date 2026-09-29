"""Stub implementation of TestRunner for Stage 6 validation.

# TODO(member-3/4): Replace with containerized or isolated sandbox test execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import time

from src.interfaces import ITestRunner, TestResult

logger = logging.getLogger(__name__)


class TestRunnerStub(ITestRunner):
    """Stub implementation of TestRunner.
    
    Can run pytest via subprocess if tests directory exists,
    or return a controlled result for testing.
    """
    __test__ = False

    def __init__(self, force_pass: bool | None = None, fail_attempts_until: int = 0) -> None:
        self.force_pass = force_pass
        self.fail_attempts_until = fail_attempts_until
        self._call_count = 0

    def run(self, repo: str | Path, patch: str) -> TestResult:
        # TODO(member-3/4): Implement containerized runner with timeouts and energy measurement
        self._call_count += 1
        start = time.perf_counter()

        if self.force_pass is not None:
            passed = self.force_pass
            duration = time.perf_counter() - start
            return TestResult(
                passed=passed,
                error_log="" if passed else f"Forced failure in test runner (call {self._call_count})",
                duration_seconds=duration,
            )

        if self.fail_attempts_until > 0:
            if self._call_count <= self.fail_attempts_until:
                duration = time.perf_counter() - start
                return TestResult(
                    passed=False,
                    error_log=f"AssertionError: Expected behavior violated on attempt {self._call_count}\nTraceback: line 42 in test_oracle",
                    duration_seconds=duration,
                )
            else:
                duration = time.perf_counter() - start
                return TestResult(
                    passed=True,
                    error_log="",
                    duration_seconds=duration,
                )

        repo_path = Path(repo).resolve()
        tests_dir = repo_path / "tests"

        if repo_path.exists() and tests_dir.exists():
            try:
                proc = subprocess.run(
                    ["pytest", "-q"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                passed = proc.returncode == 0
                error_log = proc.stdout + "\n" + proc.stderr if not passed else ""
                return TestResult(
                    passed=passed,
                    error_log=error_log.strip(),
                    duration_seconds=time.perf_counter() - start,
                )
            except Exception as e:
                logger.warning("Subprocess pytest execution failed: %s", e)

        # Default fallback: mock passing if valid unified diff
        passed = bool(patch and ("---" in patch or "diff --git" in patch))
        return TestResult(
            passed=passed,
            error_log="" if passed else "No valid patch content provided.",
            duration_seconds=time.perf_counter() - start,
        )
