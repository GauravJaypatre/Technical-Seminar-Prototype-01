# Seminar Project Findings & Synthesis Report

**Project Title:** *Benchmarking Multi-Agent Small Language Model Swarms Against Monolithic LLMs for Automated Repository Analysis*  
**Academic Context:** Final Year Computer Engineering Technical Seminar  
**Experimental Scope:** 18 Curated Micro-SWE Benchmark Repositories $\times$ 5 Random Seeds ($N=90$ runs per condition) | Authentic Three-Way Benchmark: Monolithic Groq LPU (`openai/gpt-oss-120b`, $N=90$) vs. Multi-SLM Swarms (v1 Header-Only QA vs. v2 Content-Validated QA with `deepseek-coder:6.7b` + `qwen2.5:0.5b`, $N=90$ each)

---

## Executive Summary of Empirical Findings

### 1. Authentic Three-Way Cross-System Comparison Summary (Groq LPU vs. Swarm v1 vs. Swarm v2)
*(Evaluated across all 18 Micro-SWE tasks $\times$ 5 seeds $\{42, 43, 44, 45, 46\}$, $n=30$ runs per tier, $N=90$ runs per system. Pass@1 proportions reported with 95% Wilson Score Confidence Intervals. Source: `results/prototype_run_swarm_v2_content_validated_qa/three_way_comparison_groq_v1_v2.csv`, rows 2–13).*

| System | Tier | N | Pass Count | Pass@1 (%) [95% Wilson CI] | Mean Latency (s) | Total Tokens | Fallback Rate (%) | QA Exhausted (%) | Mean QA Retries |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Groq Monolithic (120B)** | **Easy** | 30 | 22 | **73.33%** [55.55%, 85.82%] | 7.45s | 743.0 | 86.67% | 0.00% | 0.00 |
| **Groq Monolithic (120B)** | **Medium** | 30 | 27 | **90.00%** [74.38%, 96.54%] | 8.59s | 901.5 | 100.00% | 0.00% | 0.00 |
| **Groq Monolithic (120B)** | **Hard** | 30 | 28 | **93.33%** [78.68%, 98.15%] | 8.84s | 991.1 | 100.00% | 0.00% | 0.00 |
| **Groq Monolithic (120B)** | **Overall** | **90** | **77** | **85.56%** [76.84%, 91.36%] | **8.30s** | **878.5** | **95.56%** | **0.00%** | **0.00** |
| | | | | | | | | | |
| **Swarm v1 (Header-Only QA)** | **Easy** | 30 | 0 | **0.00%** [0.00%, 11.35%] | 54.69s | 946.2 | 0.00% | 0.00% | 0.03 |
| **Swarm v1 (Header-Only QA)** | **Medium** | 30 | 0 | **0.00%** [0.00%, 11.35%] | 92.90s | 1,280.2 | 0.00% | 0.00% | 0.07 |
| **Swarm v1 (Header-Only QA)** | **Hard** | 30 | 0 | **0.00%** [0.00%, 11.35%] | 116.13s | 1,380.4 | 23.33% | 0.00% | 0.00 |
| **Swarm v1 (Header-Only QA)** | **Overall** | **90** | **0** | **0.00%** [0.00%, 4.09%] | **87.91s** | **1,202.3** | **7.78%** | **0.00%** | **0.03** |
| | | | | | | | | | |
| **Swarm v2 (Content-Validated QA)** | **Easy** | 30 | 4 | **13.33%** [5.31%, 29.68%] | 104.51s | 1,461.9 | 0.00% | 70.00% | 1.50 |
| **Swarm v2 (Content-Validated QA)** | **Medium** | 30 | 0 | **0.00%** [0.00%, 11.35%] | 193.08s | 2,108.4 | 0.00% | 96.67% | 1.93 |
| **Swarm v2 (Content-Validated QA)** | **Hard** | 30 | 0 | **0.00%** [0.00%, 11.35%] | 184.71s | 1,812.6 | 16.67% | 56.67% | 1.13 |
| **Swarm v2 (Content-Validated QA)** | **Overall** | **90** | **4** | **4.44%** [1.74%, 10.88%] | **160.76s** | **1,794.3** | **5.56%** | **74.44%** | **1.52** |

### 2. Core Takeaways
1. **Frontier Baseline Dominance**: The cloud monolithic baseline (`openai/gpt-oss-120b` on Groq LPU) achieved an overall Pass@1 rate of **85.56% [95% Wilson CI: 76.84%, 91.36%]** with an average latency of **8.30s**, scaling effectively across difficulty tiers (73.33% Easy, 90.00% Medium, 93.33% Hard).
2. **Context Brittleness in Swarm v1**: Swarm v1 achieved **0.00% Pass@1 [95% Wilson CI: 0.00%, 4.09%]** because its QA agent verified only unified diff headers without checking whether the hunks matched the target files, allowing subtle context-line hallucinations to fail `git apply` in the execution sandbox.
3. **Targeted Recovery in Swarm v2**: Introducing an in-memory content-validated dry-run (`_apply_hunks_to_text`) recovered **4 successful resolutions on Easy tasks (13.33% [95% Wilson CI: 5.31%, 29.68%])**, resolving context brittleness on single-file edits.
4. **Cognitive Reasoning Ceiling**: On Medium and Hard tasks, Swarm v2 remained at **0.00% [95% Wilson CI: 0.00%, 11.35%]** despite exhausting multi-turn QA retries (74.44% exhaustion overall, 96.67% on Medium). Multi-file dependencies and complex algorithms represent a cognitive ceiling that syntax repair feedback cannot bridge for 6.7B SLMs.

---

## Direct Alignment with Seminar Objectives (O1–O4)

### Objective O1: Infrastructure & Experimental Harness
- Extended benchmarking infrastructure to log granular JSON and CSV records after every individual run (`results/prototype_run_*/`).
- Implemented a secure, isolated execution sandbox (`src/sandbox.py`) using `tempfile.TemporaryDirectory` with process timeout enforcement ($15.0\text{s}$) and robust unified diff application.
- **Harness Verification & Patch Integrity:** Audited and resolved diff application edge cases to ensure unapplicable patches are strictly rejected (`patch_applied = False`), preventing false-positive scoring of unpatched bugs.
- Synthesized and verified 18 self-contained "Micro-SWE" repositories (6 Easy, 6 Medium, 6 Hard) exhibiting 100% test failures in unpatched states and 100% test passage under ground-truth patches across 36 unit tests.

### Objective O2: Iteration Dynamics & QA Verifier Behavior
*(Evaluated on live Swarm v1 and Swarm v2 execution records across 90 runs per system).*

1. **Swarm v1 (Header-Only QA): Passive Feedback Failure**:
   - In Swarm v1, the QA verifier checked only whether the model's output contained diff headers (`diff --git`, `--- a/`, `+++ b/`).
   - Because the Code Analyzer almost always generated header-conformant text, the QA verifier triggered retries on only **2 out of 90 runs (mean 0.03 retries per run)**.
   - However, because hunk context lines had subtle line-number drifts or hallucinated variable names, `git apply` failed in the sandbox, yielding 0/90 pass rate across all tiers.
2. **Swarm v2 (Content-Validated QA): Active Iterative Reflection**:
   - Swarm v2 implemented an in-memory structural dry-run (`_apply_hunks_to_text`), comparing the analyzer's proposed diff against the exact target file contents in the sandbox before invoking git apply.
   - When context lines or offsets did not match, the verifier rejected the diff and fed the exact failure line number back to the Code Analyzer for a retry.
   - This converted the QA agent into an active reflective loop:
     - **Easy Tier:** Mean **1.50 retries per run**, with a **70.00% QA retry exhaustion rate**. This feedback loop successfully recovered **4 solves (13.33%)** on `easy_04` (seed 45) and `easy_05` (seeds 42, 44, 46).
     - **Medium Tier:** Mean **1.93 retries per run**, with a **96.67% QA retry exhaustion rate**. Despite extensive multi-turn retries, 0/30 runs passed.
     - **Hard Tier:** Mean **1.13 retries per run**, with a **56.67% QA retry exhaustion rate**. 0/30 runs passed.
3. **Dynamics Takeaway**: Iterative reflection is highly effective when the failure mode is **syntactic or contextual** (aligning diff context lines with file text on Easy tasks), but degrades into repeated retry exhaustion when the underlying failure is **semantic or algorithmic** (Medium and Hard tasks).

### Objective O3: Complexity Ceiling & Economic/Latency Break-Even (RQ2)

#### Authentic Cross-System Evaluation: Groq Monolithic LPU vs. Multi-SLM Swarm
The completed empirical benchmark compares a commercial frontier model (`openai/gpt-oss-120b` executed on Groq LPU hardware) against the local multi-agent SLM swarm (`deepseek-coder:6.7b` + `qwen2.5:0.5b` executed via local Ollama).

#### Economic Break-Even & Cost-vs-Capability Disclosure
- **Monetary Outlay vs. Latency Trade-Off**:
  - The local SLM swarm incurs **$0.00 in direct cloud API billing**, operating on local host hardware without per-token SaaS invoices.
  - However, this zero monetary outlay is paired with a substantial **~19x latency penalty**: **160.76s average wall-clock latency per run** for Swarm v2 versus **8.30s per run** for the Groq Monolithic baseline ($\Delta_{\text{lat}} = +152.47\text{s}$, Welch's $t = 21.885, p < 10^{-36}$, Cohen's $d = +3.262$).
  - *Local Compute & Energy Disclosure*: While direct monetary billing was $0.00, local host compute and electrical power consumption (CPU/GPU wattage over a 4.0-hour 90-task sweep) were not directly metered. Therefore, "zero cloud billing" must not be conflated with zero operational expense.
- **Frontier LPU Economic Profile**: Groq LPU inference completed 90 runs at high throughput (8.30s mean latency, consuming 878.5 total tokens/run) with high reliability, demonstrating an enterprise SaaS profile with predictable per-call token pricing.

#### Latency Dynamics & Complexity Ceiling Assessment
- Across all evaluated difficulty tiers, authentic empirical execution with a 6.7B parameter local model executing a multi-turn QA loop on CPU demonstrates a substantial wall-clock latency disparity compared to LPU execution:
  - **Easy tier:** $104.51\text{s}$ (Swarm v2) vs $7.45\text{s}$ (Groq) — **14.0x slower**.
  - **Medium tier:** $193.08\text{s}$ (Swarm v2) vs $8.59\text{s}$ (Groq) — **22.5x slower**.
  - **Hard tier:** $184.71\text{s}$ (Swarm v2) vs $8.84\text{s}$ (Groq) — **20.9x slower**.
- **Operational Complexity Ceiling**: A clear operational ceiling emerges. As task complexity scales to Medium and Hard, Swarm v2 latency expands to over 3 minutes per defect attempt, driven by repeated QA retries (averaging 1.93 retries on Medium with 96.67% exhaustion) without yielding successful patch resolutions.

### Objective O4: Accuracy Parity & Statistical Hypothesis Testing (RQ3)

#### Authentic Pairwise Statistical Comparison: Swarm v2 vs. Groq Monolithic LPU
*(Source: `results/prototype_run_swarm_v2_content_validated_qa/swarm_v2_vs_groq_comparison_v2_stats.csv`, rows 2–5).*

To evaluate binary Pass@1 outcomes under small sample sizes ($n=30$ per tier) and zero-count cells (Medium and Hard v2 = 0/30), statistical significance was computed using proportion-appropriate methods: **Fisher's exact test** (primary), **two-proportion $z$-test** (secondary check), and **Wilson score confidence intervals**. Effect size for accuracy is represented via **Risk Difference** ($\Delta = p_{\text{v2}} - p_{\text{Groq}}$ with 95% Newcombe Hybrid Score CI) and **Fisher Odds Ratio** (reporting both raw and Haldane-Anscombe corrected ratios), while continuous **Cohen's $d$** is retained exclusively for wall-clock latency:

| Tier | Groq N | Swarm v2 N | Groq Pass@1 (%) [95% Wilson CI] | Swarm v2 Pass@1 (%) [95% Wilson CI] | Risk Difference (% pts) [95% Newcombe CI] | Fisher OR (Raw) | Fisher OR (HA Corrected) | Fisher $p$-value (Acc) | Two-Prop $z$-stat | Two-Prop $p$-value (Acc) | Delta Latency (s) | Cohen's $d$ (Lat) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Easy** | 30 | 30 | 73.33% [55.55%, 85.82%] | **13.33%** [5.31%, 29.68%] | **-60.00%** [-74.84%, -35.85%] | 0.0559 | 0.0642 | $4.83 \times 10^{-6}$ | -4.689 | $2.74 \times 10^{-6}$ | +97.06s | +3.600 |
| **Medium** | 30 | 30 | 90.00% [74.38%, 96.54%] | **0.00%** [0.00%, 11.35%] | **-90.00%** [-96.54%, -70.69%] | 0.0000* | **0.0021** | $9.23 \times 10^{-14}$ | -7.006 | $2.44 \times 10^{-12}$ | +184.49s | +5.139 |
| **Hard** | 30 | 30 | 93.33% [78.68%, 98.15%] | **0.00%** [0.00%, 11.35%] | **-93.33%** [-98.15%, -74.79%] | 0.0000* | **0.0014** | $8.39 \times 10^{-15}$ | -7.246 | $4.30 \times 10^{-13}$ | +175.86s | +3.738 |
| **Overall** | **90** | **90** | **85.56%** [76.84%, 91.36%] | **4.44%** [1.74%, 10.88%] | **-81.11%** [-87.51%, -70.28%] | **0.0079** | **0.0091** | **$2.28 \times 10^{-31}$** | **-10.937** | **$7.67 \times 10^{-28}$** | **+152.47s** | **+3.262** |

> [!NOTE]
> **Haldane-Anscombe Correction for Zero-Count Cells**:  
> In $2 \times 2$ contingency tables where an outcome cell contains zero observations (specifically, Swarm v2's 0/30 successes on Medium and Hard tiers), the raw sample odds ratio $\frac{a \cdot d}{b \cdot c}$ degenerates to 0.0000. To resolve this degeneracy, a standard **Haldane-Anscombe correction** was applied by adding $0.5$ to all four table cells ($\tilde{a} = a + 0.5, \tilde{b} = b + 0.5, \tilde{c} = c + 0.5, \tilde{d} = d + 0.5$). This yields well-defined, non-degenerate finite odds ratios of **0.0021** on Medium and **0.0014** on Hard, confirming that the odds of defect resolution under Swarm v2 are approximately 0.21% and 0.14% of the odds under the Groq monolithic baseline.
>
> *(Supplementary Reference: Welch's $t$-test on Pass@1 as an uncorrected linear proxy yields Overall $t = -18.778, p < 10^{-39}$; Easy $t = -5.793, p < 10^{-6}$; Medium $t = -16.155, p < 10^{-15}$; Hard $t = -20.149, p < 10^{-17}$, consistent with the exact non-parametric tests).*

#### Calibrated Research Insights & Mechanistic Findings
1. **Easy-Tier Recovery and Wilson Score Confidence Interval Interpretation**:
   * On the Easy tier, Swarm v2 achieved a Pass@1 resolution rate of **13.33% [95% Wilson CI: 5.31%, 29.68%]**, resting on exactly **4 successes out of 30 runs** (tasks `easy_04` and `easy_05`).
   * *Methodological Sensitivity Note*: Because this estimate rests on 4 successful runs across $n=30$, a single-task outcome shift changes the point estimate by several percentage points. The wide confidence interval ($5.31\%$ to $29.68\%$) underscores that this recovery must be interpreted with caution as an exploratory finding rather than a definitive capability plateau.
2. **Mechanistic Separation: Context-Anchoring Brittleness vs. Cognitive Reasoning Ceilings**:
   * In Swarm v1 (Header-Only QA), the verifier evaluated only diff headers and target file presence, allowing subtle context line hallucinations and line offset drifts to slip through and fail sandbox diff application (0.00% Pass@1).
   * In Swarm v2 (Content-Validated QA), adding an in-memory structural dry-run (`_apply_hunks_to_text`) with line-accurate error feedback provided actionable critique to the Analyzer. This recovered 4/30 solves on Easy tasks, providing clear evidence that **context-anchoring brittleness** was the primary failure mode on simple, single-file defects.
   * On **Medium and Hard tiers**, however, Swarm v2 remained at **0.00%** (0/30 on Medium, 0/30 on Hard) despite 137 triggered QA retries and a **74.44% overall QA retry-exhaustion rate** (reaching 96.67% on Medium).
   * *Calibrated Causal Claim*: This empirical divergence is consistent with, and provides strong evidence for, a **cognitive reasoning capability gap** beyond what mechanical diff-repair can fix. Specifically, for this model pairing (`deepseek-coder:6.7b` + `qwen2.5:0.5b`) on the 18-task Micro-SWE suite ($n=30$ per tier), multi-file dependency tracking and algorithmic invariant synthesis require cognitive depth that small quantized models cannot overcome through diff syntax correction alone. We hedge this finding as an empirical observation scoped to this specific architecture, rather than a general claim about all SLM swarms or 6.7B models broadly.

---

## Methodological Integrity & Quarantine Disclosure

### Phase 1 Offline Simulation Bug & Artifact Quarantine

During the initial repository commit (`54dbfef`, 2026-09-08), early scaffold test runs were checked in under `results/runs/benchmark_*_20260907_*.json`. Forensic audit revealed that when Ollama was unavailable, the code fell back to an internal simulation function (`_analyze_offline` in `src/code_analyzer.py`) that rolled pseudo-random numbers against hardcoded probability arrays and directly injected `TASK_METADATA[task_id]["ground_truth_patch"]` on "success".

**Remediation & Quarantine Protocol**:
1. All 22 files in `results/runs/` dated 2026-09-07 and downstream tables in `results/tables/` are permanently quarantined.
2. They are preserved in `results/runs/README.md` strictly as forensic artifacts documenting the simulation bug for the paper's methodology and limitations sections.
3. An automated pre-commit audit tool (`analysis/verify_documentation_claims.py`) was introduced to enforce:
   - Authenticity gates (rejecting sub-2s CPU runs, canned template diffs, and zero-token logs).
   - Forbidden claim filtering (blocking references to the synthetic simulation numbers).
   - Claim-to-artifact matching (verifying every reported percentage against verified CSV records).
4. All empirical conclusions of this technical seminar rest solely on the verified live execution sweeps in `results/prototype_run_*/`.

---

## Generated Artifacts & Replication Files

1. **Master Statistical Comparison Files**:
   - `results/prototype_run_swarm_v2_content_validated_qa/swarm_v2_vs_groq_comparison_v2_stats.csv` *(Proportion-appropriate inference: Fisher exact test, two-proportion $z$-test, Wilson score CIs, Risk Difference with Newcombe CIs, Haldane-Anscombe ORs, continuous latency Cohen's $d$)*.
   - `results/prototype_run_swarm_v2_content_validated_qa/three_way_comparison_groq_v1_v2.csv` *(Consolidated 3-way benchmark comparison with Wilson score 95% CIs)*.
   - `results/prototype_run_swarm_v2_content_validated_qa/swarm_v2_vs_groq_comparison.csv` *(Original comparison file preserved for reference)*.
2. **Run Workbooks & Granular Execution Logs**:
   - `results/prototype_run_groq/Prototype_Results_Groq.xlsx` & `benchmark_prototype_monolithic_groq_20260909_164450.csv` *(Groq Monolithic baseline, N=90)*.
   - `results/prototype_run_swarm_v1_header_only_qa/benchmark_prototype_swarm_20260909_175738.csv` *(Swarm v1 baseline, N=90)*.
   - `results/prototype_run_swarm_v2_content_validated_qa/Prototype_Results_Swarm.xlsx` & `benchmark_prototype_swarm_20260909_204053.csv` *(Swarm v2 baseline, N=90)*.
3. **Quarantined Phase 1 Simulation Scaffold**:
   - `results/runs/README.md` *(Forensic documentation and quarantine manifest for 2026-09-07 scaffold runs)*.
4. **Automated Verification & Integrity Scripts**:
   - `analysis/verify_documentation_claims.py` *(Pre-commit claim-to-artifact verification tool)*.
   - `analysis/compare_swarm_vs_monolithic.py` *(Statistical test reproduction script)*.
