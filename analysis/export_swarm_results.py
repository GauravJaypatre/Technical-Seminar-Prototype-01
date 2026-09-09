"""
Multi-SLM Swarm Pipeline Exporter:
Processes results from results/prototype_run_swarm/ to produce:
1. Prototype_Results_Swarm.xlsx (Sheets: Swarm_Summary, Per_Task_Breakdown, Raw_Run_Data)
2. prototype_per_task_breakdown.csv
3. Canonical figures in results/prototype_run_swarm/figures/ only (no duplicate root files):
   - fig_prototype_accuracy_swarm.png
   - fig_prototype_latency_swarm.png
"""
import os
import sys
import glob
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SWARM_DIR = os.path.join(BASE_DIR, "results", "prototype_run_swarm")


def export_swarm_results(csv_path: str = None, output_dir: str = None):
    target_dir = output_dir or DEFAULT_SWARM_DIR
    if not os.path.isabs(target_dir):
        target_dir = os.path.join(BASE_DIR, target_dir)

    fig_dir = os.path.join(target_dir, "figures")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    if not csv_path:
        pattern = os.path.join(target_dir, "benchmark_prototype_*.csv")
        files = glob.glob(pattern)
        if not files:
            files = glob.glob(os.path.join(target_dir, "benchmark_*.csv"))
        if not files:
            raise FileNotFoundError(f"No swarm benchmark CSV found in {target_dir}")
        csv_path = max(files, key=os.path.getmtime)
    else:
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(BASE_DIR, csv_path)

    print(f"\n========================================================")
    print(f"Exporting Swarm Prototype Results & Figures")
    print(f"Source CSV:  {os.path.relpath(csv_path, BASE_DIR)}")
    print(f"Target Dir:  {os.path.relpath(target_dir, BASE_DIR)}")
    print(f"Figures Dir: {os.path.relpath(fig_dir, BASE_DIR)}")
    print(f"========================================================\n")

    df = pd.read_csv(csv_path)

    system_name = df["system"].iloc[0] if "system" in df.columns else "multi_slm_swarm"
    model_profile = df["model_profile"].iloc[0] if "model_profile" in df.columns else "deepseek-coder:6.7b+qwen2.5:0.5b"
    total_runs = len(df)
    pass_rate = df["final_pass"].mean() * 100
    mean_latency = df["cumulative_wall_clock_sec"].mean()
    mean_prompt_tokens = df["total_prompt_tokens"].mean()
    mean_completion_tokens = df["total_completion_tokens"].mean()
    total_cost = df["estimated_cost_usd"].sum()

    fallback_rate = (df["used_fallback"].mean() * 100) if "used_fallback" in df.columns else 0.0
    qa_exhaust_rate = (df["qa_exhausted"].mean() * 100) if "qa_exhausted" in df.columns else 0.0
    mean_qa_retries = df["qa_retries"].mean() if "qa_retries" in df.columns else 0.0

    print(f"Swarm Run Summary:")
    print(f"  System:                    {system_name}")
    print(f"  Model Profile:             {model_profile}")
    print(f"  Total Runs Evaluated:      {total_runs}")
    print(f"  Pass@1 Accuracy:           {pass_rate:.1f}%")
    print(f"  Fallback Usage Rate:       {fallback_rate:.1f}%")
    print(f"  QA Exhaustion Rate:        {qa_exhaust_rate:.1f}%")
    print(f"  Mean QA Retries / Task:    {mean_qa_retries:.2f}")
    print(f"  Mean Latency per Run:      {mean_latency:.2f}s")
    print(f"  Mean Prompt Tokens:        {mean_prompt_tokens:.0f}")
    print(f"  Mean Completion Tokens:    {mean_completion_tokens:.0f}")
    print(f"  Total Estimated Cost:      ${total_cost:.5f}")

    # Sheet 1: Summary Data
    summary_data = [
        {"Metric": "System", "Value": system_name},
        {"Metric": "Model Profile", "Value": model_profile},
        {"Metric": "Provider", "Value": "ollama_local"},
        {"Metric": "Total Runs Evaluated", "Value": total_runs},
        {"Metric": "Pass@1 Accuracy (%)", "Value": round(pass_rate, 2)},
        {"Metric": "Fallback Usage Rate (%)", "Value": round(fallback_rate, 2)},
        {"Metric": "QA Exhaustion Rate (%)", "Value": round(qa_exhaust_rate, 2)},
        {"Metric": "Mean QA Retries", "Value": round(mean_qa_retries, 2)},
        {"Metric": "Mean Latency (sec)", "Value": round(mean_latency, 2)},
        {"Metric": "Mean Prompt Tokens", "Value": round(mean_prompt_tokens, 1)},
        {"Metric": "Mean Completion Tokens", "Value": round(mean_completion_tokens, 1)},
        {"Metric": "Total Estimated Cost ($)", "Value": round(total_cost, 5)},
    ]
    df_summary = pd.DataFrame(summary_data)

    # Sheet 2: Per-Task Breakdown
    agg_dict = {
        "Pass_at_1": ("pass_at_1", "mean"),
        "Final_Pass": ("final_pass", "mean"),
        "Mean_Latency_Sec": ("cumulative_wall_clock_sec", "mean"),
        "Mean_Prompt_Tokens": ("total_prompt_tokens", "mean"),
        "Mean_Completion_Tokens": ("total_completion_tokens", "mean"),
        "Total_Runs": ("run_id", "count")
    }
    if "used_fallback" in df.columns:
        agg_dict["Fallback_Rate"] = ("used_fallback", "mean")
    if "qa_exhausted" in df.columns:
        agg_dict["QA_Exhausted_Rate"] = ("qa_exhausted", "mean")
    if "qa_retries" in df.columns:
        agg_dict["Mean_QA_Retries"] = ("qa_retries", "mean")

    per_task = df.groupby(["task_id", "difficulty"]).agg(**agg_dict).reset_index()
    per_task = per_task.round(3)

    # Write Excel: Prototype_Results_Swarm.xlsx
    excel_path = os.path.join(target_dir, "Prototype_Results_Swarm.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Swarm_Summary", index=False)
            per_task.to_excel(writer, sheet_name="Per_Task_Breakdown", index=False)
            df.to_excel(writer, sheet_name="Raw_Run_Data", index=False)
        print(f"Exported Excel workbook: {os.path.relpath(excel_path, BASE_DIR)}")
    except Exception as e:
        print(f"Warning writing Excel: {e}")

    # CSV breakdown
    summary_csv = os.path.join(target_dir, "prototype_per_task_breakdown.csv")
    per_task.to_csv(summary_csv, index=False)
    print(f"Exported CSV breakdown:  {os.path.relpath(summary_csv, BASE_DIR)}")

    # -----------------------------------------------------------------
    # Generate Figures (in figures/ canonical directory only)
    # -----------------------------------------------------------------
    # Figure 1: Accuracy by Task
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tasks = per_task["task_id"].tolist()
    accuracies = (per_task["Final_Pass"] * 100).tolist()
    colors = ["#2ca02c" if acc > 0 else "#d62728" for acc in accuracies]

    bars = ax.bar(tasks, accuracies, color=colors, alpha=0.85, width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Pass@1 Rate (%)")
    ax.set_title(f"Multi-SLM Swarm Benchmark: Pass@1 Accuracy by Task\n({model_profile})")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.xticks(rotation=30, ha="right")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    fig1_path = os.path.join(fig_dir, "fig_prototype_accuracy_swarm.png")
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Generated Figure 1:      {os.path.relpath(fig1_path, BASE_DIR)}")

    # Figure 2: Mean Latency by Task
    fig, ax = plt.subplots(figsize=(8, 4.5))
    latencies = per_task["Mean_Latency_Sec"].tolist()
    ax.bar(tasks, latencies, color="#9467bd", alpha=0.85, width=0.5)
    ax.set_ylabel("Wall-Clock Latency (seconds)")
    ax.set_title(f"Multi-SLM Swarm Benchmark: Mean Latency per Task\n({model_profile})")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.xticks(rotation=30, ha="right")

    fig2_path = os.path.join(fig_dir, "fig_prototype_latency_swarm.png")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Generated Figure 2:      {os.path.relpath(fig2_path, BASE_DIR)}")

    print(f"\nSwarm downstream export pipeline completed successfully!")
    return excel_path, fig1_path, fig2_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Swarm Prototype Results")
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to swarm CSV")
    parser.add_argument("--output-dir", default=None, help="Target output directory")
    args = parser.parse_args()

    export_swarm_results(csv_path=args.csv_path, output_dir=args.output_dir)
