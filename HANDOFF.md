# AERS-SLM: Member 1 Integration & Handoff Guide

## 1. Important Environment & Path Warning

### Current Workspace Location:
`c:\Users\HP\OneDrive\المستندات\Projects\TS New`

### Identified Risks:
1. **OneDrive Synchronization**:
   OneDrive continuously watches and syncs local files. During active model downloading (~1.95 GB GGUF) and git worktree operations, file locks from OneDrive can lead to `PermissionError` or corrupted locks.
2. **Non-ASCII Path Characters (`المستندات`)**:
   The Arabic characters in the user documents directory can trigger `UnicodeEncodeError: 'charmap' codec can't encode characters` in default Windows terminal codepages (`cp1252`). While Member 1 scripts explicitly reconfigure `sys.stdout` to UTF-8 with error handlers, external third-party tools might not.

### Proposed Safe Location:
`C:\dev\aers-slm` or `C:\Projects\aers-slm`  
*(Note: As instructed, no files have been moved automatically.)*

---

## 2. Merging into the Team's Real Repository

Because this local repository was initialized freshly with `git init` and currently has no remote, it should be merged into the team's upstream repository using git's `--allow-unrelated-histories` flag.

### Step-by-Step Merge Procedure:

1. **Add the team repository remote**:
   ```bash
   git remote add origin <TEAM_REPO_URL>
   git fetch origin
   ```

2. **Merge the upstream main/master into your feature branch**:
   ```bash
   git checkout feature/member1-slm-pipeline
   git merge origin/main --allow-unrelated-histories -m "chore: merge upstream main into member 1 feature branch"
   ```

3. **Expected Conflict Areas & Resolution Guidance**:
   - `src/model/`: **Owned by Member 1**. Keep Member 1's `llama_runtime.py` (singleton CPU runtime) and `inference.py` (constrained JSON decoding & usage tracking).
   - `src/agents/`: **Owned by Member 1**. Keep Member 1's specialized agents and prompts (`profiler.py`, `localizer.py`, `hypothesis.py`, `patch_generator.py`, `prompts.py`).
   - `src/schemas.py`: **Shared contracts**. Contains `RequirementContract` (strict 5 fields) and `EvidenceBaton` (<= 1,200 token hard cap, B0 <= 500 tokens). Do NOT modify field names or remove the `project()` budget enforcement method.
   - `src/interfaces.py`: **Cross-member contracts**. Defines `typing.Protocol`s for Members 2, 3, and 4.
   - `src/stubs/`: **Temporary placeholders**. Real implementations by Members 2, 3, and 4 will replace these stubs via `src/registry.py`.

4. **Verify tests pass after merge**:
   ```bash
   pytest -v
   ```

5. **Push feature branch to remote**:
   ```bash
   git push -u origin feature/member1-slm-pipeline
   ```

---

## 3. Component Ownership & Integration Table

| Module / Component | Owner | Interface Protocol | Status |
|-------------------|-------|--------------------|--------|
| **SLM Runtime** (`src/model/`) | **Member 1** | `InferenceEngine` | Complete (CPU GGUF singleton, memory monitor) |
| **Agents & Prompts** (`src/agents/`) | **Member 1** | `BaseAgent` | Complete (Stages 1, 3, 4, 6) |
| **Pipeline Orchestrator** (`src/pipeline.py`) | **Member 1** | `AERSPipeline` | Complete (6 sequential stages, retry loop, rollback) |
| **Repository Profiler** | Member 2 | `IRepositoryProfiler` | Stub ready (`src/stubs/profiler_stub.py`) |
| **Hybrid Context Compiler** | Member 2 | `IContextCompiler` | Stub ready (`src/stubs/context_compiler_stub.py`) |
| **Evidence Gate & Critic** | Member 3 | `IEvidenceGate`, `IFalsificationCritic` | Stub ready (`src/stubs/evidence_gate_stub.py`) |
| **Test Runner** | Member 3/4 | `ITestRunner` | Stub ready (`src/stubs/test_runner_stub.py`) |
| **Sandbox & Rollback** | Member 4 | `IRepoSandbox` | Stub ready (`src/stubs/sandbox_stub.py`) |
| **Usage & Metrics Tracking** | Member 4 | `UsageTracker` | Complete (Tokens, latency, peak RSS) |

---

## 4. Real Model Weights Check

The model file is stored at:
`models/qwen2.5-coder-3b-instruct-q4_k_m.gguf` (~1.96 GB)

To download or verify model presence:
```bash
python scripts/download_model.py
```
