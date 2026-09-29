"""Component registry for dependency injection.

Allows Member 2, Member 3, and Member 4 to plug in their production modules
without requiring modifications to pipeline orchestration code.
"""

from __future__ import annotations

import logging
from typing import Type

from src.interfaces import (
    IContextCompiler,
    IEvidenceGate,
    IFalsificationCritic,
    IRepoSandbox,
    IRepositoryProfiler,
    ITestLogCorrelator,
    ITestRunner,
)
from src.stubs.context_compiler_stub import ContextCompilerStub
from src.stubs.correlator_stub import TestLogCorrelatorStub
from src.stubs.evidence_gate_stub import EvidenceGateStub, FalsificationCriticStub
from src.stubs.profiler_stub import RepositoryProfilerStub
from src.stubs.sandbox_stub import RepoSandboxStub
from src.stubs.test_runner_stub import TestRunnerStub

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Registry managing instantiated components for the AERS-SLM pipeline."""

    def __init__(
        self,
        profiler: IRepositoryProfiler | None = None,
        context_compiler: IContextCompiler | None = None,
        evidence_gate: IEvidenceGate | None = None,
        critic: IFalsificationCritic | None = None,
        test_runner: ITestRunner | None = None,
        sandbox: IRepoSandbox | None = None,
        correlator: ITestLogCorrelator | None = None,
    ) -> None:
        self.profiler: IRepositoryProfiler = profiler or RepositoryProfilerStub()
        self.context_compiler: IContextCompiler = context_compiler or ContextCompilerStub()
        self.critic: IFalsificationCritic = critic or FalsificationCriticStub()
        self.evidence_gate: IEvidenceGate = evidence_gate or EvidenceGateStub(critic=self.critic)
        self.test_runner: ITestRunner = test_runner or TestRunnerStub()
        self.sandbox: IRepoSandbox = sandbox or RepoSandboxStub()
        self.correlator: ITestLogCorrelator = correlator or TestLogCorrelatorStub()

    def register_profiler(self, profiler: IRepositoryProfiler) -> None:
        """Register Member 2 Repository Profiler."""
        self.profiler = profiler
        logger.info("Registered custom RepositoryProfiler: %s", type(profiler).__name__)

    def register_context_compiler(self, compiler: IContextCompiler) -> None:
        """Register Member 2 Context Compiler."""
        self.context_compiler = compiler
        logger.info("Registered custom ContextCompiler: %s", type(compiler).__name__)

    def register_evidence_gate(self, gate: IEvidenceGate) -> None:
        """Register Member 3 Evidence Gate."""
        self.evidence_gate = gate
        logger.info("Registered custom EvidenceGate: %s", type(gate).__name__)

    def register_critic(self, critic: IFalsificationCritic) -> None:
        """Register Member 3 Falsification Critic."""
        self.critic = critic
        logger.info("Registered custom FalsificationCritic: %s", type(critic).__name__)

    def register_test_runner(self, runner: ITestRunner) -> None:
        """Register Member 3/4 Test Runner."""
        self.test_runner = runner
        logger.info("Registered custom TestRunner: %s", type(runner).__name__)

    def register_sandbox(self, sandbox: IRepoSandbox) -> None:
        """Register Member 4 Repo Sandbox."""
        self.sandbox = sandbox
        logger.info("Registered custom RepoSandbox: %s", type(sandbox).__name__)

    def register_correlator(self, correlator: ITestLogCorrelator) -> None:
        """Register Test Log Correlator."""
        self.correlator = correlator
        logger.info("Registered custom TestLogCorrelator: %s", type(correlator).__name__)
