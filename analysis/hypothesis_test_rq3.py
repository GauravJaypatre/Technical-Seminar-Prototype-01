"""
Statistical Analysis for RQ3:
Swarm Pass@1 vs. Swarm Final Pass (Pass@K) vs. Monolithic Pass@1.

Computes:
1. Swarm Pass@1 (initial one-shot accuracy)
2. Swarm Final Pass (accuracy after K=5 self-correction iterations)
3. Self-Correction Yield: Delta_K = Pass@K - Pass@1
4. Monolithic Pass@1
5. Exact McNemar's Test on paired binary outcomes (Swarm Final Pass vs Monolithic Pass@1)
6. Paired Student's t-test on task-level mean accuracy across seeds (n=18)
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_rq3_analysis(swarm_csv: str = None, mono_csv: str = None):
    if not swarm_csv:
        files = [f for f in glob.glob(os.path.join(BASE_DIR, "results", "runs", "benchmark_swarm_*.csv")) if "ablate" not in f and os.path.getsize(f) > 1000]
        if not files:
            raise FileNotFoundError("No full swarm benchmark CSV found.")
        swarm_csv = max(files, key=os.path.getmtime)

    if not mono_csv:
        files = [f for f in glob.glob(os.path.join(BASE_DIR, "results", "runs", "benchmark_monolithic_*.csv")) if os.path.getsize(f) > 1000]
        if not files:
            raise FileNotFoundError("No monolithic benchmark CSV found.")
        mono_csv = max(files, key=os.path.getmtime)

    print(f"Executing Hypothesis Tests (RQ3):")
    print(f"  Swarm Source:      {os.path.relpath(swarm_csv, BASE_DIR)}")
    print(f"  Monolithic Source: {os.path.relpath(mono_csv, BASE_DIR)}")

    df_swarm = pd.read_csv(swarm_csv)
    df_mono = pd.read_csv(mono_csv)

    # Task-level mean accuracy across seeds (n=18)
    swarm_task_acc = df_swarm.groupby(["task_id", "difficulty"]).agg(
        swarm_pass1=("pass_at_1", "mean"),
        swarm_final=("final_pass", "mean")
    ).reset_index()

    mono_task_acc = df_mono.groupby(["task_id", "difficulty"]).agg(
        mono_pass1=("final_pass", "mean")
    ).reset_index()

    merged = pd.merge(swarm_task_acc, mono_task_acc, on=["task_id", "difficulty"])
    merged["self_correction_yield"] = merged["swarm_final"] - merged["swarm_pass1"]

    tiers = ["easy", "medium", "hard"]
    summary_rows = []

    for tier in tiers:
        sub = merged[merged["difficulty"] == tier]
        sw_p1 = sub["swarm_pass1"].mean()
        sw_pk = sub["swarm_final"].mean()
        yield_k = sw_pk - sw_p1
        mo_p1 = sub["mono_pass1"].mean()

        # Paired t-test within tier
        if np.all(sub["swarm_final"] == sub["mono_pass1"]):
            t_stat, p_val_t = 0.0, 1.0
        else:
            t_res = stats.ttest_rel(sub["swarm_final"], sub["mono_pass1"])
            t_stat, p_val_t = t_res.statistic, t_res.pvalue

        summary_rows.append({
            "Tier": tier.capitalize(),
            "Swarm_Pass@1": round(sw_p1, 3),
            "Swarm_Final_Pass@K": round(sw_pk, 3),
            "Self_Correction_Gain": round(yield_k, 3),
            "Monolithic_Pass@1": round(mo_p1, 3),
            "Paired_t_stat": round(t_stat, 3),
            "p_value_paired_t": round(p_val_t, 4)
        })

    # Overall Metrics (n=18 tasks)
    ov_sw_p1 = merged["swarm_pass1"].mean()
    ov_sw_pk = merged["swarm_final"].mean()
    ov_yield = ov_sw_pk - ov_sw_p1
    ov_mo_p1 = merged["mono_pass1"].mean()
    t_res_all = stats.ttest_rel(merged["swarm_final"], merged["mono_pass1"])

    summary_rows.append({
        "Tier": "Overall (N=18)",
        "Swarm_Pass@1": round(ov_sw_p1, 3),
        "Swarm_Final_Pass@K": round(ov_sw_pk, 3),
        "Self_Correction_Gain": round(ov_yield, 3),
        "Monolithic_Pass@1": round(ov_mo_p1, 3),
        "Paired_t_stat": round(t_res_all.statistic, 3),
        "p_value_paired_t": round(t_res_all.pvalue, 4)
    })

    summary_df = pd.DataFrame(summary_rows)
    print("\n--- RQ3 Accuracy & Self-Correction Decomposition Table ---")
    print(summary_df.to_string(index=False))

    # =========================================================================
    # EXACT MCNEMAR'S TEST ON PAIRED TASK OUTCOMES
    # Dichotomized: Task resolved (>= 50% of seeds passed) vs unresolved
    # =========================================================================
    merged["swarm_resolved"] = merged["swarm_final"] >= 0.5
    merged["mono_resolved"] = merged["mono_pass1"] >= 0.5

    # 2x2 Contingency Table
    both_pass = sum(merged["swarm_resolved"] & merged["mono_resolved"])
    swarm_only = sum(merged["swarm_resolved"] & ~merged["mono_resolved"])
    mono_only = sum(~merged["swarm_resolved"] & merged["mono_resolved"])
    both_fail = sum(~merged["swarm_resolved"] & ~merged["mono_resolved"])

    table = [[both_pass, swarm_only], [mono_only, both_fail]]
    mcnemar_res = mcnemar(table, exact=True)

    print("\n--- McNemar's Test on Paired Task Outcomes (N=18) ---")
    print(f"Contingency Matrix (2x2):")
    print(f"  Both Passed:               {both_pass}")
    print(f"  Swarm Passed, Mono Failed: {swarm_only}")
    print(f"  Mono Passed, Swarm Failed: {mono_only}")
    print(f"  Both Failed:               {both_fail}")
    print(f"McNemar exact p-value: {mcnemar_res.pvalue:.4f}")

    out_path = os.path.join(BASE_DIR, "results", "tables", "rq3_hypothesis_test.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"\nRQ3 summary exported to: {os.path.relpath(out_path, BASE_DIR)}")
    return summary_df

if __name__ == "__main__":
    s_path = sys.argv[1] if len(sys.argv) > 1 else None
    m_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_rq3_analysis(s_path, m_path)
