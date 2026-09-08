"""
Statistical Analysis for RQ1:
Does mean iteration-count-to-match differ significantly by task difficulty (Easy vs Medium vs Hard)?

CRITICAL METHODOLOGICAL SAFEGUARD:
This script explicitly aggregates runs across seeds to ONE mean-iterations-to-solve
value per task (n=18, with n=6 per difficulty tier) before executing ANOVA and
Kruskal-Wallis. Non-independent repeated seeds (n=54) are NOT treated as independent rows,
strictly preventing pseudoreplication.

Extends: reference_slm_study/Data Analysis and Supported Scripts/Scripts/anova_rq1.py and tdhf.py.
Adds: Eta-squared, Kruskal-Wallis H-test, Epsilon-squared, and Cliff's Delta.
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def calculate_cliffs_delta(x, y):
    """Computes Cliff's Delta non-parametric effect size between two groups."""
    n_x = len(x)
    n_y = len(y)
    more = sum(1 for i in x for j in y if i > j)
    less = sum(1 for i in x for j in y if i < j)
    return (more - less) / (n_x * n_y) if (n_x * n_y) > 0 else 0.0

def run_rq1_analysis(csv_file_path: str = None) -> pd.DataFrame:
    # Locate latest full swarm CSV (excluding ablation files) if not explicitly passed
    if not csv_file_path:
        search_pattern = os.path.join(BASE_DIR, "results", "runs", "benchmark_swarm_*.csv")
        files = [f for f in glob.glob(search_pattern) if "ablate" not in f]
        if not files:
            raise FileNotFoundError("No full swarm benchmark CSV found in results/runs/.")
        csv_file_path = max(files, key=os.path.getmtime)

    print(f"Loading swarm benchmark data from: {os.path.relpath(csv_file_path, BASE_DIR)}")
    df = pd.read_csv(csv_file_path)

    # If task failed after K iterations, iterations_to_success is recorded as 6 (K+1 penalty)
    # for ordinal ranking in non-parametric tests
    df["iterations_effective"] = df["iterations_to_success"].apply(
        lambda x: 6 if (pd.isna(x) or str(x).lower() == "none" or str(x).strip() == "") else float(x)
    )

    # =========================================================================
    # STEP 1: TASK-LEVEL AGGREGATION (PREVENT PSEUDOREPLICATION)
    # n = 18 independent tasks (6 Easy, 6 Medium, 6 Hard)
    # =========================================================================
    task_agg = df.groupby(["task_id", "difficulty"]).agg(
        mean_iterations=("iterations_effective", "mean"),
        pass_rate=("final_pass", "mean"),
        mean_latency=("cumulative_wall_clock_sec", "mean"),
        mean_tokens=("total_completion_tokens", "mean")
    ).reset_index()

    print(f"\nAggregated {len(df)} seed-level runs into {len(task_agg)} independent task-level observations.")
    tier_counts = task_agg["difficulty"].value_counts().to_dict()
    print(f"Sample sizes per difficulty tier: {tier_counts} (Total N = {len(task_agg)})")

    # Group slices
    easy_iters = task_agg[task_agg["difficulty"] == "easy"]["mean_iterations"].values
    med_iters = task_agg[task_agg["difficulty"] == "medium"]["mean_iterations"].values
    hard_iters = task_agg[task_agg["difficulty"] == "hard"]["mean_iterations"].values

    # Descriptive Statistics
    desc = task_agg.groupby("difficulty")["mean_iterations"].agg(["count", "mean", "std", "median"])
    desc["IQR"] = task_agg.groupby("difficulty")["mean_iterations"].apply(lambda s: np.percentile(s, 75) - np.percentile(s, 25))
    print("\n--- Descriptive Statistics (Task-Level Mean Iterations to Solve) ---")
    print(desc.round(3))

    # =========================================================================
    # STEP 2: ONE-WAY ANOVA & ETA-SQUARED
    # =========================================================================
    f_stat, p_val_anova = stats.f_oneway(easy_iters, med_iters, hard_iters)
    
    # Calculate Eta-squared: SS_between / SS_total
    all_iters = task_agg["mean_iterations"].values
    grand_mean = np.mean(all_iters)
    ss_total = np.sum((all_iters - grand_mean) ** 2)
    ss_between = (
        len(easy_iters) * (np.mean(easy_iters) - grand_mean) ** 2 +
        len(med_iters) * (np.mean(med_iters) - grand_mean) ** 2 +
        len(hard_iters) * (np.mean(hard_iters) - grand_mean) ** 2
    )
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    print("\n--- One-Way ANOVA Results ---")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value:     {p_val_anova:.4e}")
    print(f"Eta-squared (eta^2 effect size): {eta_squared:.4f}")

    # =========================================================================
    # STEP 3: TUKEY HSD POST-HOC TEST
    # =========================================================================
    tukey = pairwise_tukeyhsd(
        endog=task_agg["mean_iterations"],
        groups=task_agg["difficulty"],
        alpha=0.05
    )
    print("\n--- Tukey HSD Post-Hoc Test (Pairwise Differences) ---")
    print(tukey)

    # =========================================================================
    # STEP 4: NON-PARAMETRIC KRUSKAL-WALLIS & EPSILON-SQUARED (ROBUSTNESS CHECK)
    # =========================================================================
    h_stat, p_val_kruskal = stats.kruskal(easy_iters, med_iters, hard_iters)
    
    # Epsilon-squared: (H - k + 1) / (n - k)
    n_total = len(task_agg)
    k_groups = 3
    epsilon_squared = (h_stat - k_groups + 1) / (n_total - k_groups) if (n_total - k_groups) > 0 else 0.0

    print("\n--- Non-Parametric Kruskal-Wallis H-Test (Robust to Censoring) ---")
    print(f"H-statistic:       {h_stat:.4f}")
    print(f"p-value:           {p_val_kruskal:.4e}")
    print(f"Epsilon-squared (eps^2 effect size): {epsilon_squared:.4f}")

    # Pairwise Cliff's Delta
    delta_em = calculate_cliffs_delta(med_iters, easy_iters)
    delta_eh = calculate_cliffs_delta(hard_iters, easy_iters)
    delta_mh = calculate_cliffs_delta(hard_iters, med_iters)

    print("\n--- Pairwise Non-Parametric Effect Sizes (Cliff's Delta) ---")
    print(f"Easy vs. Medium: delta = {delta_em:+.3f}")
    print(f"Easy vs. Hard:   delta = {delta_eh:+.3f}")
    print(f"Medium vs. Hard: delta = {delta_mh:+.3f}")

    # Save summary table
    summary_path = os.path.join(BASE_DIR, "results", "tables", "rq1_anova_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df = pd.DataFrame([{
        "Test": "One-Way ANOVA",
        "Statistic_Name": "F",
        "Statistic_Value": round(f_stat, 4),
        "p_value": round(p_val_anova, 6),
        "Effect_Size_Name": "Eta-squared",
        "Effect_Size_Value": round(eta_squared, 4),
        "Sample_Size_N": n_total,
        "Aggregation_Level": "Task-Level Mean (n=18)"
    }, {
        "Test": "Kruskal-Wallis",
        "Statistic_Name": "H",
        "Statistic_Value": round(h_stat, 4),
        "p_value": round(p_val_kruskal, 6),
        "Effect_Size_Name": "Epsilon-squared",
        "Effect_Size_Value": round(epsilon_squared, 4),
        "Sample_Size_N": n_total,
        "Aggregation_Level": "Task-Level Mean (n=18)"
    }])
    summary_df.to_csv(summary_path, index=False)
    print(f"\nRQ1 statistical summary exported to: {os.path.relpath(summary_path, BASE_DIR)}")
    return task_agg

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_rq1_analysis(csv_file)
