"""
Prototype Pipeline Exporter:
Processes results from prototype runs (results/prototype_run/ or results/prototype_run_groq/)
to validate the complete downstream flow:
Sandbox/Diff Harness -> CSV -> Excel (openpyxl) -> Figures (matplotlib).

Guarantees:
- Never touches results/runs/
- Never overwrites results/tables/Results.xlsx
- Never touches results/figures/
- Never modifies PROVISIONAL markers in SEMINAR_SYNTHESIS.md
"""
import os
import sys
import glob
import json
import argparse
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def export_prototype_results(csv_path: str = None, output_dir: str = None):
    # 1. Resolve output directory and CSV path
    if output_dir:
        target_dir = output_dir if os.path.isabs(output_dir) else os.path.join(BASE_DIR, output_dir)
    elif csv_path:
        target_dir = os.path.dirname(os.path.abspath(csv_path))
    else:
        target_dir = os.path.join(BASE_DIR, "results", "prototype_run")

    fig_dir = os.path.join(target_dir, "figures")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    if not csv_path:
        pattern = os.path.join(target_dir, "benchmark_prototype_*.csv")
        files = glob.glob(pattern)
        if not files:
            files = glob.glob(os.path.join(target_dir, "benchmark_*.csv"))
        if not files:
            raise FileNotFoundError(f"No prototype benchmark CSV found in {target_dir}")
        csv_path = max(files, key=os.path.getmtime)
    else:
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(BASE_DIR, csv_path)

    print(f"\n========================================================")
    print(f"Exporting Prototype Results & Figures")
    print(f"Source CSV:  {os.path.relpath(csv_path, BASE_DIR)}")
    print(f"Target Dir:  {os.path.relpath(target_dir, BASE_DIR)}")
    print(f"Figures Dir: {os.path.relpath(fig_dir, BASE_DIR)}")
    print(f"========================================================\n")

    df = pd.read_csv(csv_path)

    # Validate prototype tagging
    if "is_prototype" in df.columns:
        all_proto = df["is_prototype"].all()
        print(f"Validation: All records tagged is_prototype = {all_proto}")

    provider = df["provider"].iloc[0] if "provider" in df.columns else "unknown"
    model_name = df["model_name"].iloc[0] if "model_name" in df.columns else "unknown"
    total_runs = len(df)
    pass_rate = df["final_pass"].mean() * 100
    mean_latency = df["cumulative_wall_clock_sec"].mean()
    mean_prompt_tokens = df["total_prompt_tokens"].mean()
    mean_completion_tokens = df["total_completion_tokens"].mean()
    total_cost = df["estimated_cost_usd"].sum()

    fallback_rate = (df["used_fallback"].mean() * 100) if "used_fallback" in df.columns else 0.0

    print(f"Prototype Run Summary:")
    print(f"  Provider:                  {provider}")
    print(f"  Model:                     {model_name}")
    print(f"  Total Runs Evaluated:      {total_runs}")
    print(f"  Pass@1 Accuracy:           {pass_rate:.1f}%")
    print(f"  Fallback Usage Rate:       {fallback_rate:.1f}%")
    print(f"  Mean Latency per Run:      {mean_latency:.2f}s")
    print(f"  Mean Prompt Tokens:        {mean_prompt_tokens:.0f}")
    print(f"  Mean Completion Tokens:    {mean_completion_tokens:.0f}")
    print(f"  Total Estimated Cost:      ${total_cost:.5f}")

    # Sheet 1: Summary Stats
    summary_data = [{
        "Metric": "Provider", "Value": provider
    }, {
        "Metric": "Model Name", "Value": model_name
    }, {
        "Metric": "Total Runs Evaluated", "Value": total_runs
    }, {
        "Metric": "Pass@1 Accuracy (%)", "Value": round(pass_rate, 2)
    }, {
        "Metric": "Fallback Usage Rate (%)", "Value": round(fallback_rate, 2)
    }, {
        "Metric": "Mean Latency (sec)", "Value": round(mean_latency, 2)
    }, {
        "Metric": "Mean Prompt Tokens", "Value": round(mean_prompt_tokens, 1)
    }, {
        "Metric": "Mean Completion Tokens", "Value": round(mean_completion_tokens, 1)
    }, {
        "Metric": "Total Estimated Cost ($)", "Value": round(total_cost, 5)
    }]
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

    per_task = df.groupby(["task_id", "difficulty"]).agg(**agg_dict).reset_index()
    per_task = per_task.round(3)

    # Write Excel file(s)
    excel_filename = "Prototype_Results_Groq.xlsx" if provider.lower() == "groq" else "Prototype_Results.xlsx"
    excel_path = os.path.join(target_dir, excel_filename)
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Prototype_Summary", index=False)
            per_task.to_excel(writer, sheet_name="Per_Task_Breakdown", index=False)
            df.to_excel(writer, sheet_name="Raw_Run_Data", index=False)
        print(f"Exported Excel workbook: {os.path.relpath(excel_path, BASE_DIR)}")

        # Also write standard Prototype_Results.xlsx in target_dir if different
        if excel_filename != "Prototype_Results.xlsx":
            alt_path = os.path.join(target_dir, "Prototype_Results.xlsx")
            shutil.copyfile(excel_path, alt_path)
    except Exception as e:
        print(f"Warning writing Excel: {e}")

    # Also save CSV summary
    summary_csv = os.path.join(target_dir, "prototype_per_task_breakdown.csv")
    per_task.to_csv(summary_csv, index=False)
    print(f"Exported CSV breakdown:  {os.path.relpath(summary_csv, BASE_DIR)}")

    # -----------------------------------------------------------------
    # Generate Prototype Figures
    # -----------------------------------------------------------------
    # Figure 1: Per-Task Pass Rate
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tasks = per_task["task_id"].tolist()
    accuracies = (per_task["Final_Pass"] * 100).tolist()
    colors = ["#2ca02c" if acc > 0 else "#d62728" for acc in accuracies]

    bars = ax.bar(tasks, accuracies, color=colors, alpha=0.85, width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Pass@1 Rate (%)")
    ax.set_title(f"Prototype Benchmark: Pass@1 Accuracy by Task\n({provider} - {model_name})")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.xticks(rotation=30, ha="right")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    fig1_name = "fig_prototype_accuracy_groq.png" if provider.lower() == "groq" else "fig_prototype_accuracy.png"
    fig1_path = os.path.join(fig_dir, fig1_name)
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Generated Figure 1:      {os.path.relpath(fig1_path, BASE_DIR)}")

    # Figure 2: Latency per Task
    fig, ax = plt.subplots(figsize=(8, 4.5))
    latencies = per_task["Mean_Latency_Sec"].tolist()
    ax.bar(tasks, latencies, color="#1f77b4", alpha=0.85, width=0.5)
    ax.set_ylabel("Wall-Clock Latency (seconds)")
    ax.set_title(f"Prototype Benchmark: Mean Latency per Task\n({provider} - {model_name})")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.xticks(rotation=30, ha="right")

    fig2_name = "fig_prototype_latency_groq.png" if provider.lower() == "groq" else "fig_prototype_latency.png"
    fig2_path = os.path.join(fig_dir, fig2_name)
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Generated Figure 2:      {os.path.relpath(fig2_path, BASE_DIR)}")

    print(f"\nPrototype downstream pipeline completed successfully!")
    return excel_path, fig1_path, fig2_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export prototype benchmark results to Excel and Figures")
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to prototype CSV file")
    parser.add_argument("--output-dir", default=None, help="Target directory for outputs")
    args = parser.parse_args()

    export_prototype_results(csv_path=args.csv_path, output_dir=args.output_dir)
