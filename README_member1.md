# AERS-SLM: Member 1 - Core SLM Pipeline & Runtime

> **Adaptive Evidence-Routed Sequential SLM Intelligence for Repository-Level Software Engineering**  
> Component: Member 1 (SLM Runtime, Agents, Prompts, Requirement Contract & Pipeline Orchestrator)

---

## 1. Overview & Architectural Principles

AERS-SLM enables a single, shared 3B small language model (`Qwen2.5-Coder-3B-Instruct`, Q4_K_M GGUF, ~1.95 GB) to perform end-to-end repository-level bug fixing on a CPU-only 16 GB laptop using `llama.cpp`.

Key architectural invariants maintained by Member 1:
1. **Single Model Instance (Singleton)**: Exactly one model instance is loaded into system RAM via `LlamaRuntime`. Stages are specialized strictly through system prompts, never by spinning up multiple models.
2. **Bounded Inter-Stage Evidence Baton**: Inter-stage communication is conducted exclusively through a typed JSON `EvidenceBaton` strictly capped at **<= 1,200 tokens** (initial baton $B_0 \le 500$ tokens).
3. **Stateless Stages**: Conversational history, reasoning traces, and unverified chain-of-thought are wiped clean between stages. Each agent receives only the bounded baton and compiled context.
4. **Deterministic & CPU-Only**: Zero cloud API dependencies. Run with temperature `0.2`, fixed seed `42`, and `n_gpu_layers=0`. Memory consumption stays well under 12 GB RAM.
5. **Test-Driven Self-Correction**: Stage 6 executes an automated retry loop (max 2 retries) with failing test tracebacks appended to the baton and automatic sandbox rollback upon test failure.

---

## 2. Directory Layout

```
src/
 ├── model/
 │    ├── __init__.py
 │    ├── llama_runtime.py       # Thread-safe GGUF singleton runtime (CPU-only)
 │    └── inference.py           # generate(), generate_json(), streaming, FakeLLM, UsageTracker
 ├── agents/
 │    ├── __init__.py
 │    ├── base.py                # BaseAgent: zero-history execution contract
 │    ├── profiler.py            # Stage 1: Repo Profile & Requirement Contract (B0 <= 500 tokens)
 │    ├── localizer.py           # Stage 3: Fault Localizer (top k<=5 files & ranges, hallucination filter)
 │    ├── hypothesis.py          # Stage 4: Root Cause Analyst (<=3 falsifiable hypotheses)
 │    ├── patch_generator.py     # Stage 6: Minimal unified diff generator + retry handler
 │    └── prompts.py             # Curated system prompts (<400 tokens each)
 ├── stubs/                      # Functional stubs for Member 2, 3, and 4
 │    ├── __init__.py
 │    ├── profiler_stub.py       # RepositoryProfilerStub (Member 2)
 │    ├── context_compiler_stub.py # ContextCompilerStub (Member 2)
 │    ├── evidence_gate_stub.py  # EvidenceGateStub & FalsificationCriticStub (Member 3)
 │    ├── test_runner_stub.py    # TestRunnerStub (Member 3/4)
 │    ├── sandbox_stub.py        # RepoSandboxStub (Member 4)
 │    └── correlator_stub.py     # TestLogCorrelatorStub (Member 2/3)
 ├── schemas.py                  # Pydantic schemas (RequirementContract, EvidenceBaton, etc.)
 ├── interfaces.py               # Typing Protocols for cross-member contracts
 ├── registry.py                 # Dependency injection component registry
 ├── config.py                   # Centralized configuration with env/YAML overrides
 └── pipeline.py                 # AERSPipeline: 6-stage sequential orchestrator
scripts/
 ├── download_model.py           # Downloads Qwen2.5-Coder-3B Q4_K_M GGUF from HF Hub
 └── run_pipeline.py             # CLI entrypoint for running the pipeline
tests/
 ├── conftest.py                 # Shared fixtures and mock environment
 ├── test_schemas.py             # Contract validation, baton caps & projection tests
 ├── test_runtime.py             # JSON repair, context guard, usage metrics tests
 ├── test_agents.py              # Agent logic, hallucination filtering, hypothesis caps
 └── test_pipeline.py            # End-to-end pipeline execution with stubs & retries
```

---

## 3. Installation & Setup

### 3.1 Requirements
- Python 3.10+
- Git
- Recommended: C++ Build Tools (MSVC on Windows / GCC or Clang on Linux/macOS) for compiling `llama-cpp-python`

### 3.2 Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Or install in editable mode
pip install -e .
```

### 3.3 Installing `llama-cpp-python` (CPU Only)

For pre-built CPU binaries or compilation:
```bash
# Windows / Linux / macOS CPU build
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

*Note:* You can run tests and dry-run execution immediately with `FakeLLM` even without `llama-cpp-python` installed.

---

## 4. Downloading the Model

Run the download script to fetch the ~1.95 GB quantized GGUF model into `models/`:

```bash
python scripts/download_model.py
```

Options:
- `--dest <path>`: Specify destination directory (default: `models`)
- `--force`: Force re-download even if the file exists

---

## 5. Configuration Options

All options can be set via environment variables (prefixed with `AERS_`) or in a YAML configuration file.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `models/qwen2.5-coder-3b-instruct-q4_k_m.gguf` | Path to Qwen2.5-Coder GGUF file |
| `n_ctx` | `8192` | Model context window size (tokens) |
| `n_threads` | Physical CPU cores | Thread pool size for CPU execution |
| `n_batch` | `512` | Prompt processing batch size |
| `temperature` | `0.2` | Sampling temperature |
| `top_p` | `0.95` | Nucleus sampling cutoff |
| `seed` | `42` | Fixed seed for reproducible generation |
| `use_mmap` | `True` | Memory-map weights from disk |
| `use_mlock` | `False` | Pin model memory in RAM |
| `n_gpu_layers` | `0` | Strictly CPU-only (0 GPU layers) |
| `context_budget` | `4000` | Stage 2 context token budget |
| `baton_max_tokens` | `1200` | EvidenceBaton hard token cap |
| `initial_baton_max_tokens` | `500` | Budget for initial Baton B0 |
| `max_retries` | `2` | Number of Stage 6 test-driven retries |
| `dry_run` | `False` | Run with FakeLLM mock engine |
| `log_dir` | `logs` | Structured JSONL logging directory |

---

## 6. Running the CLI

### 6.1 Dry-Run / Test Mode (no weights required)

```bash
python scripts/run_pipeline.py --repo . --issue "Fix division by zero in divide()" --dry-run
```

### 6.2 Running with Real GGUF Model

```bash
python scripts/run_pipeline.py --repo path/to/target/repo --issue "Fix KeyError in user authentication session handler"
```

You can also pass a file containing the issue description:
```bash
python scripts/run_pipeline.py --repo path/to/target/repo --issue path/to/issue.txt
```

---

## 7. How to Plug in Member 2, 3, and 4 Modules

Member 1 provides clear `typing.Protocol` interfaces in `src/interfaces.py` and a dependency injection `ComponentRegistry` in `src/registry.py`.

### Member 2: Repository Profiler & Context Compiler
Implement `IRepositoryProfiler` and `IContextCompiler`:
```python
from src.interfaces import IRepositoryProfiler, IContextCompiler, RepoProfile
from src.registry import ComponentRegistry
from src.pipeline import AERSPipeline

class ProductionContextCompiler(IContextCompiler):
    def compile(self, contract, repo, budget=4000) -> str:
        # Your BM25 / AST symbol-graph retrieval here
        return "... compiled slice ..."

registry = ComponentRegistry()
registry.register_context_compiler(ProductionContextCompiler())

pipeline = AERSPipeline(registry=registry)
```

### Member 3: Evidence Gate, Critic & Test Runner
Implement `IEvidenceGate`, `IFalsificationCritic`, and `ITestRunner`:
```python
from src.interfaces import IEvidenceGate, ITestRunner, TestResult, GateStatus

class ProductionEvidenceGate(IEvidenceGate):
    def evaluate(self, baton, repo) -> tuple[Hypothesis, GateStatus]:
        # Deterministic verification & type checking
        ...

registry.register_evidence_gate(ProductionEvidenceGate())
```

### Member 4: Sandbox & Metrics Tracking
Implement `IRepoSandbox`:
```python
from src.interfaces import IRepoSandbox

class WorktreeSandbox(IRepoSandbox):
    def prepare(self, repo): ...
    def apply_patch(self, patch: str) -> bool: ...
    def rollback(self): ...

registry.register_sandbox(WorktreeSandbox())
```

Member 4 can also read token usage and latency metrics directly via `pipeline.engine.tracker.summary()`.

---

## 8. Verification & Running Tests

Run the unit test suite without needing model weights:
```bash
pytest -v
```

Run including slow tests (with real GGUF weights if present):
```bash
pytest -v -m "slow"
```

Structured pipeline execution events are logged to `logs/pipeline_<timestamp>.jsonl` with token counts, execution duration, and stage outcomes.
