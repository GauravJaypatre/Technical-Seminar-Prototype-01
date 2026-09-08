# Benchmarking Multi-Agent Small Language Model Swarms Against Monolithic LLMs for Automated Repository Analysis

**Final Year Computer Engineering Technical Seminar**  
**Author:** Gaurav Jaypatre  
**Institution:** Department of Computer Engineering  

---

## 📌 Executive Summary

This research benchmarks an autonomous, local swarm of quantized **Small Language Models (SLMs)**—orchestrated via a sequential multi-agent architecture—against a **monolithic frontier LLM (Claude 3.5 Sonnet)** for automated repository-level defect resolution.

Using a curated suite of **18 self-contained "Micro-SWE" benchmark repositories** spanning three difficulty tiers (Easy, Medium, Hard), we evaluate:
1. **Iteration Dynamics (RQ1):** How test-driven self-correction iteration counts scale as task complexity increases.
2. **Complexity Ceiling & Economics (RQ2):** The economic break-even and latency boundary where multi-agent reflection stops being cost-effective compared to single-shot frontier inference.
3. **Reasoning Parity via Self-Correction (RQ3):** Whether multi-turn execution feedback allows small models (1.3B–1.5B parameters) to match or exceed frontier LLMs in real bug-fixing accuracy.
4. **Router Agent Ablation:** Whether top-level architectural routing adds value on simple single-file repairs or merely introduces latency and token overhead.

---

## 🏗️ System Architecture

The local SLM swarm pipeline operates entirely on commodity hardware (16 GB RAM) through sequential multi-agent execution with model offloading (`keep_alive: 0`):

```
                       +---------------------------------------+
                       |              Task Issue               |
                       |       (Repository & Test Suite)       |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |             Router Agent              |
                       |    (Qwen2.5-Coder:1.5B-Instruct)      |
                       |  - Fault localization & strategy      |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |          Code Analyzer Agent          |
                       |   (DeepSeek-Coder:1.3B-Instruct)      |
                       |  - Generates unified diff patch       |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |         Test-Driven Sandbox           |
                       |  - Ephemeral directory replication    |
                       |  - Strict unified diff verification   |
                       |  - Isolated pytest execution (15s TO) |
                       +---------------------------------------+
                                 /                   \
                       [Tests Pass]                 [Tests Fail]
                            |                             |
                            v                             v
                       +----------+        +-------------------------------+
                       | Success  |        |       QA Verifier Agent       |
                       | (Exit)   |        | (Qwen2.5-Coder:1.5B-Instruct) |
                       +----------+        | - Analyzes pytest traceback   |
                                           | - Formulates targeted advice  |
                                           +-------------------------------+
                                                          |
                                                          +---> [Loop back to Analyzer, K <= 5]
```

---

## 📊 Core Empirical Findings

### 1. Validated Internal Swarm Finding (Core Result)
- **Single-Shot Pass@1:** The local SLM swarm achieves a **37.0% Pass@1 resolution rate** (20/54 runs) without feedback.
- **Iterative Pass@K ($K \le 5$):** With test-driven execution feedback, the swarm reaches a final **Pass@K resolution rate of 98.1%** (53/54 runs).
- **Self-Correction Yield:** Multi-turn reflection provides a **+61.1 percentage point gain** ($\Delta_K = +61.1\%$), proving that iterative execution feedback bridges the reasoning gap of small quantized models.

### 2. Iteration Scaling by Difficulty Tier (RQ1)
- Evaluated on **task-level aggregated means ($n=18$, with $n=6$ per tier)** to strictly prevent pseudoreplication across repeated random seeds ($S \in \{42, 43, 44\}$):
  - **Easy:** $\mu = 1.28 \pm 0.14$ iterations (Median: 1.33, IQR: 0.00)
  - **Medium:** $\mu = 2.06 \pm 0.33$ iterations (Median: 2.17, IQR: 0.58)
  - **Hard:** $\mu = 3.06 \pm 0.33$ iterations (Median: 3.00, IQR: 0.00)
- **One-Way ANOVA:** $F = 61.27, p = 6.06 \times 10^{-8}, \eta^2 = 0.8909$ (89.1% of iteration variance is explained by difficulty).
- **Tukey HSD Post-Hoc:** All pairwise differences are statistically significant ($p < 0.001$).
- **Non-Parametric Robustness:** Kruskal-Wallis $H = 15.73, p = 0.00038, \epsilon^2 = 0.9151$. Pairwise Cliff's Delta equals $+1.000$ across all pairs due to strict non-overlapping separation:
  $$\max(\text{Easy}) = 1.333 < \min(\text{Medium}) = 1.667 < \max(\text{Medium}) = 2.333 < \min(\text{Hard}) = 2.667$$

### 3. Router Agent Ablation (Easy Tier)
- On single-file Easy tasks, both **Full Swarm (Router Active)** and **Swarm-NoRouter (Direct Pass)** tied at **1.28 mean iterations**.
- **Degenerate Wilcoxon Test:** Across all 6 tasks and 3 seeds, iteration counts were identical ($d_i = 0$), rendering the Wilcoxon signed-rank test degenerate ($W = 0, p = \text{N/A}$).
- **Efficiency Finding:** Bypassing the router saved **25.7 completion tokens per run** ($95.5$ vs $121.2$ tokens) and **0.09s** dispatch latency, demonstrating that top-level routing should be bypassed for trivial single-file edits.

### 4. Complexity Ceiling & Economic Break-Even (RQ2)
- **Inference Cost:** Under an operational private compute proxy ($0.50 per 1M tokens), the swarm cost remains $34\times–55\times$ cheaper than commercial frontier SaaS retail pricing.
- **Latency Ratio:** Swarm wall-clock execution ($1.62\text{s}$ Easy, $1.89\text{s}$ Medium, $2.29\text{s}$ Hard) maintains a narrow $1.13\times–1.41\times$ ratio against monolithic baselines, indicating no decisive latency ceiling for interactive developer workflows.

---

## 📁 Repository Structure

```
.
├── analysis/                     # Statistical analysis and figure generation scripts
│   ├── anova_rq1.py              # One-Way ANOVA, Kruskal-Wallis, Tukey HSD, Cliff's Delta
│   ├── complexity_ceiling_rq2.py # Cost and latency break-even analysis
│   ├── hypothesis_test_rq3.py    # McNemar test & paired t-test
│   ├── router_ablation.py        # Easy-tier router ablation analysis
│   ├── plot_curves.py            # Generates Figure 1 and Figure 2
│   └── export_master_results.py  # Exports multi-sheet Results.xlsx and per_task_results.csv
├── data/
│   ├── mock_repos/               # 18 curated Micro-SWE benchmark repositories
│   │   ├── easy_01 ... easy_06   # Single-file logic, validation, edge cases
│   │   ├── med_01 ... med_06     # State machines, caching, concurrent buffers
│   │   └── hard_01 ... hard_06   # Parsers, compilers, expression evaluators
│   └── task_manifest.json        # Benchmark registry with metadata & ground truths
├── results/
│   ├── figures/                  # Publication-ready plots (PDF & PNG)
│   │   ├── fig1_iteration_curve.pdf / .png
│   │   └── fig2_complexity_ceiling.pdf / .png
│   ├── tables/                   # Statistical summary CSVs & master Excel workbook
│   │   ├── Results.xlsx          # 5-sheet master workbook
│   │   ├── per_task_results.csv
│   │   ├── rq1_anova_summary.csv
│   │   ├── rq2_complexity_ceiling.csv
│   │   ├── rq3_hypothesis_test.csv
│   │   └── router_ablation_summary.csv
│   ├── runs/                     # Granular JSON and CSV logs for all experimental sweeps
│   └── SEMINAR_SYNTHESIS.md      # Comprehensive technical seminar report
├── scripts/
│   ├── audit_failures.py         # Automated failure diagnostic tool
│   └── test_harness_cases.py     # Rigorous test suite for diff and sandbox edge cases
├── src/
│   ├── benchmarking_extended.py  # Core logging & resource monitoring infrastructure
│   ├── code_analyzer.py          # Code Analyzer agent (DeepSeek-Coder)
│   ├── monolithic_baseline.py    # Monolithic frontier baseline (Claude 3.5 Sonnet)
│   ├── qa_verifier.py            # QA Verifier agent (Qwen2.5-Coder)
│   ├── router.py                 # Router agent (Qwen2.5-Coder)
│   ├── sandbox.py                # Isolated execution sandbox with diff engine
│   └── swarm_pipeline.py         # Multi-turn sequential swarm orchestrator
├── tests/
│   └── test_mock_repos.py        # 36 unit tests validating all 18 mock repositories
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Reproduction & Usage Guide

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
# Verify sandbox and diff edge cases (4/4 must pass)
python scripts/test_harness_cases.py

# Verify ground-truth patches across all 18 repositories (36/36 must pass)
python -m pytest tests/test_mock_repos.py
```

### 3. Execute Swarm Benchmark Sweeps
```bash
# Full Swarm sweep (18 tasks x 3 seeds = 54 runs)
python src/swarm_pipeline.py --seeds 42 43 44

# Router ablation sweep on Easy tier (6 tasks x 3 seeds = 18 runs)
python src/swarm_pipeline.py --ablate_router_easy --seeds 42 43 44
```

### 4. Execute Statistical Analysis & Plotting
```bash
# Run RQ1 ANOVA and non-parametric tests
python analysis/anova_rq1.py

# Run Router Ablation analysis
python analysis/router_ablation.py

# Run RQ2 Complexity Ceiling analysis
python analysis/complexity_ceiling_rq2.py

# Run RQ3 Hypothesis testing
python analysis/hypothesis_test_rq3.py

# Generate publication figures
python analysis/plot_curves.py

# Export master Excel workbook (Results.xlsx)
python analysis/export_master_results.py
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
