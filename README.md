# Benchmarking Multi-Agent Small Language Model Swarms Against Monolithic LLMs for Automated Repository Analysis

**Final Year Computer Engineering Technical Seminar**  
**Author:** Gaurav Jaypatre  
**Institution:** Department of Computer Engineering  

---

## 📌 Executive Summary

This research investigates whether an autonomous, local multi-agent swarm of quantized **Small Language Models (SLMs)**—orchestrated sequentially with iterative feedback—can achieve defect resolution parity with a **monolithic frontier LLM baseline (`openai/gpt-oss-120b` executed on Groq LPU hardware)** on repository-level code repair.

Using a curated suite of **18 self-contained "Micro-SWE" benchmark repositories** spanning three difficulty tiers (Easy, Medium, Hard) across 5 random seeds ($N=90$ runs per system), we evaluate:
1. **Defect Resolution Accuracy (Pass@1):** Real patch synthesis and test passage across Easy, Medium, and Hard repository defects under isolated execution.
2. **Mechanistic Failure Modes:** The root causes of SLM repair failures, separating context-anchoring and git apply brittleness from cognitive reasoning ceilings on multi-file state.
3. **Execution Latency & Operational Economics:** The wall-clock execution time and token consumption trade-offs between local SLM multi-turn execution and high-throughput cloud LPU inference.

---

## 🏗️ System Architecture

The local SLM swarm pipeline operates on commodity hardware through sequential multi-agent execution with model offloading (`keep_alive: 0`):

```
                       +---------------------------------------------+
                       |                 Task Issue                  |
                       |          (Repository & Test Suite)          |
                       +---------------------------------------------+
                                              |
                                              v
                       +---------------------------------------------+
                       |                Router Node                  |
                       |         (Deterministic Heuristics)          |
                       |  - Zero LLM calls, zero token cost          |
                       |  - Direct vs. decomposed routing strategy   |
                       +---------------------------------------------+
                                              |
                                              v
                       +---------------------------------------------+
                       |             Code Analyzer Node              |
                       |            (deepseek-coder:6.7b)            |
                       |  - Ingests repository files & issue text    |
                       |  - Generates unified diff patch             |
                       +---------------------------------------------+
                                              |
                                              v
                       +---------------------------------------------+
                       |              QA Verifier Node               |
                       |               (qwen2.5:0.5b)                |
                       |  - In-memory structural dry-run             |
                       |  - Validates headers, hunks, context lines  |
                       |  - Inspects file existence & line offsets   |
                       +---------------------------------------------+
                                      /              \
                          [Patch Valid]              [Patch Malformed]
                                |                             |
                                |                  +-------------------------+
                                |                  | Conditional Retry Check |
                                |                  |    (retry_count < 2)    |
                                |                  +-------------------------+
                                |                     /                   \
                                |              [Under Cap]             [Exhausted]
                                |                   |                       |
                                |                   v                       |
                                |       +-----------------------+           |
                                |       | Loop back to Analyzer |           |
                                |       | with exact line-level |           |
                                |       | rejection critique    |           |
                                |       +-----------------------+           |
                                |                                           |
                                +---------------------+---------------------+
                                                      |
                                                      v
                                       +-----------------------------+
                                       |     Test-Driven Sandbox     |
                                       |      (src/sandbox.py)       |
                                       |  - Ephemeral directory clone|
                                       |  - Strict git apply patch   |
                                       |  - Isolated pytest (15s TO) |
                                       +-----------------------------+
                                                /           \
                                      [Tests Pass]         [Tests Fail]
                                           |                     |
                                           v                     v
                                    [Task Resolved]       [Task Unresolved]
```

---

## 📊 Core Empirical Findings

### 1. Authentic Three-Way Cross-System Comparison

Evaluated across all 18 Micro-SWE tasks $\times$ 5 random seeds $\{42, 43, 44, 45, 46\}$ ($n=30$ runs per tier, $N=90$ runs per system). Proportions are reported with **95% Wilson Score Confidence Intervals**:

| System | Tier | N | Pass Count | Pass@1 (%) [95% Wilson CI] | Mean Latency (s) | Total Tokens | Fallback Rate (%) | QA Exhausted (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Groq Monolithic (120B)** | Easy | 30 | 22 | **73.33%** [55.55%, 85.82%] | 7.45s | 743.0 | 86.67% | 0.00% |
| **Groq Monolithic (120B)** | Medium | 30 | 27 | **90.00%** [74.38%, 96.54%] | 8.59s | 901.5 | 100.00% | 0.00% |
| **Groq Monolithic (120B)** | Hard | 30 | 28 | **93.33%** [78.68%, 98.15%] | 8.84s | 991.1 | 100.00% | 0.00% |
| **Groq Monolithic (120B)** | **Overall** | **90** | **77** | **85.56%** [76.84%, 91.36%] | **8.30s** | **878.5** | **95.56%** | **0.00%** |
| | | | | | | | | |
| **Swarm v1 (Header-Only QA)** | Easy | 30 | 0 | **0.00%** [0.00%, 11.35%] | 54.69s | 946.2 | 0.00% | 0.00% |
| **Swarm v1 (Header-Only QA)** | Medium | 30 | 0 | **0.00%** [0.00%, 11.35%] | 92.90s | 1,280.2 | 0.00% | 0.00% |
| **Swarm v1 (Header-Only QA)** | Hard | 30 | 0 | **0.00%** [0.00%, 11.35%] | 116.13s | 1,380.4 | 23.33% | 0.00% |
| **Swarm v1 (Header-Only QA)** | **Overall** | **90** | **0** | **0.00%** [0.00%, 4.09%] | **87.91s** | **1,202.3** | **7.78%** | **0.00%** |
| | | | | | | | | |
| **Swarm v2 (Content-Validated QA)** | Easy | 30 | 4 | **13.33%** [5.31%, 29.68%] | 104.51s | 1,461.9 | 0.00% | 70.00% |
| **Swarm v2 (Content-Validated QA)** | Medium | 30 | 0 | **0.00%** [0.00%, 11.35%] | 193.08s | 2,108.4 | 0.00% | 96.67% |
| **Swarm v2 (Content-Validated QA)** | Hard | 30 | 0 | **0.00%** [0.00%, 11.35%] | 184.71s | 1,812.6 | 16.67% | 56.67% |
| **Swarm v2 (Content-Validated QA)** | **Overall** | **90** | **4** | **4.44%** [1.74%, 10.88%] | **160.76s** | **1,794.3** | **5.56%** | **74.44%** |

*(Source: `results/prototype_run_swarm_v2_content_validated_qa/three_way_comparison_groq_v1_v2.csv`, rows 2–13).*

---

### 2. Proportion-Appropriate Statistical Hypothesis Testing

Because Pass@1 outcomes are binary and small sample sizes ($n=30$ per tier) contain zero-count cells (Swarm v2 Medium and Hard = 0/30), statistical significance is evaluated using exact and proportion-appropriate tests:

| Tier | Groq Pass@1 (%) | Swarm v2 Pass@1 (%) | Risk Difference (% pts) [95% Newcombe CI] | Fisher Odds Ratio (Raw) | Fisher Odds Ratio (HA Corrected) | Fisher $p$-value | Two-Prop $z$-stat | Two-Prop $p$-value | Delta Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Easy** | 73.33% | **13.33%** | **-60.00%** [-74.84%, -35.85%] | 0.0559 | 0.0642 | $4.83 \times 10^{-6}$ | -4.689 | $2.74 \times 10^{-6}$ | +97.06s |
| **Medium** | 90.00% | **0.00%** | **-90.00%** [-96.54%, -70.69%] | 0.0000* | **0.0021** | $9.23 \times 10^{-14}$ | -7.006 | $2.44 \times 10^{-12}$ | +184.49s |
| **Hard** | 93.33% | **0.00%** | **-93.33%** [-98.15%, -74.79%] | 0.0000* | **0.0014** | $8.39 \times 10^{-15}$ | -7.246 | $4.30 \times 10^{-13}$ | +175.86s |
| **Overall** | **85.56%** | **4.44%** | **-81.11%** [-87.51%, -70.28%] | **0.0079** | **0.0091** | **$2.28 \times 10^{-31}$** | **-10.937** | **$7.67 \times 10^{-28}$** | **+152.47s** |

*(Source: `results/prototype_run_swarm_v2_content_validated_qa/swarm_v2_vs_groq_comparison_v2_stats.csv`, rows 2–5).*

* **Haldane-Anscombe Odds Ratio Correction:** Where zero cells exist (Medium and Hard tiers), adding 0.5 to all contingency cells resolves sample odds ratio degeneracy, yielding finite odds ratios of **0.0021** (Medium) and **0.0014** (Hard).
* **Easy-Tier Recovery:** In-memory structural dry-run validation in Swarm v2 recovered **4/30 solves on Easy** (13.33% [95% Wilson CI: 5.31%, 29.68%]), demonstrating that diff syntax and context-anchoring brittleness were the primary failure modes on simple single-file edits.
* **Cognitive Reasoning Ceiling:** On Medium and Hard tiers, Swarm v2 remained at **0.00%** despite 137 triggered QA retries and a **74.44% QA exhaustion rate** (reaching 96.67% on Medium). Multi-file dependencies and state invariant reasoning represent a cognitive ceiling that diff syntax repair cannot overcome for 6.7B SLMs.
* **Latency Disparity:** Swarm v2 averaged **160.76s** wall-clock execution per run on commodity CPU vs. **8.30s** for the Groq Monolithic baseline (~19x latency penalty).

---

### 3. Quarantine Notice: Phase 1 Offline Simulation Scaffold

> [!CAUTION]
> Early repository scaffold files (`results/runs/benchmark_*_20260907_*.json` and `.csv`) were generated by an **offline Monte Carlo probabilistic fallback simulator** (`_analyze_offline` in `src/code_analyzer.py`), NOT by live neural inference.
> All such files and their downstream summary tables in `results/tables/` are permanently quarantined in [`results/runs/README.md`](file:///c:/Users/HP/OneDrive/المستندات/Technical%20Seminar%20Prototype%2001/results/runs/README.md). They are preserved strictly as forensic documentation of the offline simulation bug for the paper's methodology section. All empirical findings presented in this research rest solely on the verified live execution sweeps in `results/prototype_run_*/`.

---

## 📁 Repository Structure

```
.
├── analysis/                                          # Statistical analysis and claim verification scripts
│   ├── compare_swarm_vs_monolithic.py                 # Primary statistical comparator (Fisher, Wilson CIs, HA OR)
│   ├── export_master_results.py                       # Excel export utility for benchmark runs
│   ├── export_prototype_results.py                    # Formats Groq prototype runs into Excel workbook
│   ├── export_swarm_results.py                        # Formats Swarm prototype runs into Excel workbook
│   └── verify_documentation_claims.py                 # Pre-commit automated claim-to-artifact audit tool
├── data/
│   ├── mock_repos/                                    # 18 curated Micro-SWE benchmark repositories
│   │   ├── easy_01 ... easy_06                        # Single-file logic, validation, edge cases
│   │   ├── med_01 ... med_06                          # State machines, caching, concurrent buffers
│   │   └── hard_01 ... hard_06                        # Parsers, compilers, expression evaluators
│   └── task_manifest.json                             # Benchmark registry with metadata & ground truths
├── results/
│   ├── prototype_run_groq/                            # Monolithic baseline runs (openai/gpt-oss-120b, N=90)
│   │   ├── benchmark_prototype_monolithic_groq_*.csv  # Raw per-run execution logs
│   │   └── Prototype_Results_Groq.xlsx                # Verified per-tier results workbook
│   ├── prototype_run_swarm_v1_header_only_qa/         # Swarm v1 runs (Header-Only QA, N=90)
│   │   └── benchmark_prototype_swarm_*.csv            # Raw per-run execution logs (0.00% Pass@1)
│   ├── prototype_run_swarm_v2_content_validated_qa/   # Swarm v2 runs (Content-Validated QA, N=90)
│   │   ├── three_way_comparison_groq_v1_v2.csv        # Consolidated 3-way benchmark comparison
│   │   ├── swarm_v2_vs_groq_comparison_v2_stats.csv   # Proportion-appropriate statistical test results
│   │   └── Prototype_Results_Swarm.xlsx               # Verified per-tier results workbook
│   ├── runs/                                          # QUARANTINED: Phase 1 offline simulation scaffold files
│   │   └── README.md                                  # Forensic documentation and quarantine manifest
│   ├── tables/                                        # Quarantined Phase 1 summary tables
│   └── SEMINAR_SYNTHESIS.md                           # Comprehensive technical seminar report
├── src/
│   ├── baseline_client.py                             # Provider client library (Groq API, rate limiting, token counting)
│   ├── monolithic_baseline.py                         # Live Groq monolithic benchmark harness (produces results/prototype_run_groq/; fallback neutralized)
│   ├── benchmarking_extended.py                       # Checkpointing logger and resource monitor
│   ├── sandbox.py                                     # Isolated execution sandbox with strict diff validation
│   ├── swarm/                                         # LangGraph Multi-SLM Swarm implementation
│   │   ├── agents.py                                  # RouterAgent (heuristic), CodeAnalyzerAgent, QAVerifierAgent
│   │   ├── graph.py                                   # LangGraph StateGraph workflow (retry loop <= 2)
│   │   └── swarm_runner.py                            # Benchmark execution runner for Swarm pipeline
│   ├── code_analyzer.py                               # (Phase 1 legacy scaffold - superseded by src/swarm/agents.py; fallback neutralized)
│   ├── qa_verifier.py                                 # (Phase 1 legacy scaffold - superseded by src/swarm/agents.py)
│   ├── router.py                                      # (Phase 1 legacy scaffold - superseded by src/swarm/agents.py)
│   └── swarm_pipeline.py                              # (Phase 1 legacy scaffold - superseded by src/swarm/swarm_runner.py)
├── tests/
│   ├── test_baseline_client.py                        # Tests for monolithic baseline client
│   ├── test_mock_repos.py                             # 36 unit tests validating all 18 mock repositories
│   ├── test_sandbox.py                                # Tests for execution sandbox & diff application
│   └── test_swarm.py                                  # Tests for LangGraph swarm workflow & agents
├── requirements.txt                                   # Pinned Python dependencies
└── README.md                                          # Project documentation
```

---

## 🚀 Reproduction & Verification Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/GauravJaypatre/Technical-Seminar.git
cd Technical-Seminar

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Harness & Mock Repositories
```bash
# Verify ground-truth patches across all 18 repositories (36/36 must pass)
python -m pytest tests/test_mock_repos.py
```

### 3. Run Pre-Commit Claim-to-Artifact Verification Audit
```bash
# Enforces authenticity gates and verifies all numbers match raw CSV artifacts
python analysis/verify_documentation_claims.py
```

### 4. Recompute Statistical Significance & Comparisons
```bash
# Recompute Fisher exact tests, Wilson CIs, and Haldane-Anscombe ORs
python analysis/compare_swarm_vs_monolithic.py
```

---

## 📜 Academic Citation

If referencing this codebase or experimental methodology in related research:

```bibtex
@misc{jaypatre2026benchmarking,
  author = {Jaypatre, Gaurav},
  title = {Benchmarking Multi-Agent Small Language Model Swarms Against Monolithic LLMs for Automated Repository Analysis},
  year = {2026},
  publisher = {Department of Computer Engineering},
  url = {https://github.com/GauravJaypatre/Technical-Seminar}
}
```
