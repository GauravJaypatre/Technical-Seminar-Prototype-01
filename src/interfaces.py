"""Component interfaces and Protocols for AERS-SLM.

Defines the integration boundaries between:
- Member 1: Core SLM Pipeline, Prompts, Agents, Runtime
- Member 2: Repository Profiler, AST/BM25 Context Compiler
- Member 3: Evidence Gate, Falsification Critic, Test Runner
- Member 4: Resource Tracking, Metrics, Repository Sandbox
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from src.schemas import EvidenceBaton, Hypothesis, RequirementContract, SuspiciousRange


class GateStatus(str, Enum):
    """Evaluation status returned by Stage 5 Evidence Gate."""
    SUPPORTED = "SUPPORTED"
    RESOLVED_CONFLICT = "RESOLVED_CONFLICT"
    CONFLICT = "CONFLICT"


class RepoProfile(BaseModel):
    """Deterministic structural profile of the target repository (Member 2)."""
    repo_path: str
    language: str = "python"
    build_system: str = "pip"
    test_framework: str = "pytest"
    file_tree: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CriticVerdict(BaseModel):
    """Output from the conditional falsification critic (Member 3)."""
    verdict: str = Field(description="SUPPORTED, FALSIFIED, or INCONCLUSIVE")
    reasoning: str = ""
    falsification_evidence: str = ""


class TestResult(BaseModel):
    """Result of running validation tests in the sandbox (Member 3/4)."""
    passed: bool
    error_log: str = ""
    duration_seconds: float = 0.0


@runtime_checkable
class IRepositoryProfiler(Protocol):
    """Stage 1a: Scans repository to determine language, build system, and test runner."""
    def profile(self, repo: str | Path) -> RepoProfile:
        ...


@runtime_checkable
class IContextCompiler(Protocol):
    """Stage 2: Hybrid Context Compiler (AST / BM25) bounded to <= budget tokens."""
    def compile(
        self,
        contract: RequirementContract,
        repo: str | Path,
        budget: int = 4000
    ) -> str:
        ...


@runtime_checkable
class IEvidenceGate(Protocol):
    """Stage 5: Deterministic tool-based verification and conditional falsification."""
    def evaluate(
        self,
        baton: EvidenceBaton,
        repo: str | Path
    ) -> tuple[Hypothesis, GateStatus]:
        ...


@runtime_checkable
class IFalsificationCritic(Protocol):
    """Stage 5 optional: Evaluates conflicting hypotheses to pick or refine winner."""
    def critique(self, hypothesis: Hypothesis, evidence: str) -> CriticVerdict:
        ...


@runtime_checkable
class ITestRunner(Protocol):
    """Executes the test suite against the target repository."""
    def run(self, repo: str | Path, patch: str) -> TestResult:
        ...


@runtime_checkable
class IRepoSandbox(Protocol):
    """Member 4: Sandbox isolation managing git checkout, patch application, and rollback."""
    def prepare(self, repo: str | Path) -> None:
        ...

    def apply_patch(self, patch: str) -> bool:
        ...

    def rollback(self) -> None:
        ...


@runtime_checkable
class ITestLogCorrelator(Protocol):
    """Correlates failing stack traces to AST nodes and suspicious line ranges."""
    def correlate(self, traceback: str, repo: str | Path) -> list[SuspiciousRange]:
        ...
