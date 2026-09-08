"""
Router Agent Ablation Study Analysis (Easy Tier).
Empirically tests whether the Router Agent provides value for single-file,
single-function tasks or merely introduces latency and token overhead.
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_router_ablation_analysis(full_csv: str = None, ablate_csv: str = None):
    # Locate full and ablated CSV files
    if not full_csv:
        files = [f for f in glob.glob(os.path.join(BASE_DIR, "results", "runs", "benchmark_swarm_*.csv")) if "ablate" not in f]
        if not files:
            raise FileNotFoundError("No non-ablated swarm CSV found.")
        full_csv = max(files, key=os.path.getmtime)

    if not ablate_csv:
        files = glob.glob(os.path.join(BASE_DIR, "results", "runs", "*ablate_easy*.csv"))
        if not files:
            print("No router-ablation CSV found yet. Run `python src/swarm_pipeline.py --ablate_router_easy` to generate.")
            return None
        ablate_csv = max(files, key=os.path.getmtime)

    print(f"Analyzing Router Ablation on Easy Tasks:")
    print(f"  Swarm-Full:      {os.path.relpath(full_csv, BASE_DIR)}")
    print(f"  Swarm-NoRouter:  {os.path.relpath(ablate_csv, BASE_DIR)}")

    # Support either single combined CSV with both systems or two separate CSVs
    if full_csv and (ablate_csv is None or full_csv == ablate_csv):
        df_all = pd.read_csv(full_csv)
        f_easy = df_all[(df_all["difficulty"] == "easy") & (df_all["system"] != "swarm_slm_no_router")]
        a_easy = df_all[(df_all["difficulty"] == "easy") & (df_all["system"] == "swarm_slm_no_router")]
    else:
        df_full = pd.read_csv(full_csv)
        df_ablate = pd.read_csv(ablate_csv)
        f_easy = df_full[df_full["difficulty"] == "easy"]
        a_easy = df_ablate[df_ablate["difficulty"] == "easy"]

    f_agg = f_easy.groupby("task_id").agg(
        full_iters=("iterations_to_success", lambda s: pd.to_numeric(s, errors="coerce").fillna(6).mean()),
        full_latency=("cumulative_wall_clock_sec", "mean"),
        full_tokens=("total_completion_tokens", "mean")
    ).reset_index()

    a_agg = a_easy.groupby("task_id").agg(
        ablate_iters=("iterations_to_success", lambda s: pd.to_numeric(s, errors="coerce").fillna(6).mean()),
        ablate_latency=("cumulative_wall_clock_sec", "mean"),
        ablate_tokens=("total_completion_tokens", "mean")
    ).reset_index()

    merged = pd.merge(f_agg, a_agg, on="task_id")

    print("\n--- Router Ablation Comparison (Easy Tier per Task) ---")
    print(merged.to_string(index=False))

    mean_full_iter = merged["full_iters"].mean()
    mean_abl_iter = merged["ablate_iters"].mean()
    mean_full_lat = merged["full_latency"].mean()
    mean_abl_lat = merged["ablate_latency"].mean()
    mean_full_tok = merged["full_tokens"].mean()
    mean_abl_tok = merged["ablate_tokens"].mean()

    print("\n--- Summary Metrics ---")
    print(f"Mean Iterations: Full = {mean_full_iter:.2f} vs. NoRouter = {mean_abl_iter:.2f}")
    print(f"Mean Latency:    Full = {mean_full_lat:.2f}s vs. NoRouter = {mean_abl_lat:.2f}s (Delta = {mean_full_lat - mean_abl_lat:.2f}s saved)")
    print(f"Mean Tokens:     Full = {mean_full_tok:.1f} vs. NoRouter = {mean_abl_tok:.1f}")

    if len(merged) >= 3 and not np.all(merged["full_iters"] == merged["ablate_iters"]):
        w_res = stats.wilcoxon(merged["full_iters"], merged["ablate_iters"])
        w_stat = round(float(w_res.statistic), 3)
        w_pval = round(float(w_res.pvalue), 4)
        print(f"Wilcoxon signed-rank test on iterations: W = {w_stat}, p = {w_pval}")
    else:
        w_stat = "Degenerate (W=0)"
        w_pval = "N/A (All differences = 0)"
        print("Iterations identical across conditions: Wilcoxon signed-rank test is degenerate (zero non-zero differences).")

    out_df = pd.DataFrame([{
        "Condition": "Swarm-Full (Router Active)",
        "Mean_Iterations": round(mean_full_iter, 2),
        "Mean_Latency_Sec": round(mean_full_lat, 2),
        "Mean_Completion_Tokens": round(mean_full_tok, 1),
        "Wilcoxon_Statistic": w_stat,
        "Wilcoxon_p_value": w_pval
    }, {
        "Condition": "Swarm-NoRouter (Direct Pass)",
        "Mean_Iterations": round(mean_abl_iter, 2),
        "Mean_Latency_Sec": round(mean_abl_lat, 2),
        "Mean_Completion_Tokens": round(mean_abl_tok, 1),
        "Wilcoxon_Statistic": w_stat,
        "Wilcoxon_p_value": w_pval
    }])

    out_path = os.path.join(BASE_DIR, "results", "tables", "router_ablation_summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nRouter ablation summary saved to: {os.path.relpath(out_path, BASE_DIR)}")
    return out_df

if __name__ == "__main__":
    f_p = sys.argv[1] if len(sys.argv) > 1 else None
    a_p = sys.argv[2] if len(sys.argv) > 2 else None
    run_router_ablation_analysis(f_p, a_p)
