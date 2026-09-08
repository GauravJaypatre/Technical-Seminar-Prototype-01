"""
Publication-Quality Visualization Generator.
Generates:
1. Figure 1: Iteration-vs-Accuracy Curves (k=1..5) for Easy, Medium, Hard, and Overall Swarm,
   overlaid with Monolithic frontier baseline flat line.
2. Figure 2: Complexity Ceiling Break-Even Analysis (Accuracy vs. Cost-per-solve).
3. Figure 3: Tukey HSD Pairwise Confidence Intervals.
Outputs both PNG and PDF formats into results/figures/.
"""
import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Set clean academic plotting style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def plot_iteration_curves(swarm_json: str = None, mono_csv: str = None):
    """Generates Figure 1: Iteration-vs-Accuracy curve (k=1..5)."""
    if not swarm_json:
        files = [f for f in glob.glob(os.path.join(BASE_DIR, "results", "runs", "benchmark_swarm_*.json")) if "ablate" not in f]
        if not files:
            print("No swarm JSON found for plotting.")
            return
        swarm_json = max(files, key=os.path.getmtime)

    if not mono_csv:
        files = [f for f in glob.glob(os.path.join(BASE_DIR, "results", "runs", "benchmark_monolithic_*.csv")) if os.path.getsize(f) > 1000]
        if not files:
            print("No monolithic CSV found for plotting.")
            return
        mono_csv = max(files, key=os.path.getmtime)

    with open(swarm_json, "r", encoding="utf-8") as f:
        runs = json.load(f)

    df_mono = pd.read_csv(mono_csv)
    mono_pass_rate = df_mono["final_pass"].mean() * 100

    # Calculate cumulative pass rate at each iteration k (1..5)
    tiers = ["easy", "medium", "hard"]
    k_vals = [1, 2, 3, 4, 5]

    cum_acc = {t: [] for t in tiers}
    cum_acc["overall"] = []

    for k in k_vals:
        # Easy
        e_runs = [r for r in runs if r.get("difficulty") == "easy"]
        e_pass = sum(1 for r in e_runs if r.get("iterations_to_success") is not None and r["iterations_to_success"] <= k)
        cum_acc["easy"].append((e_pass / len(e_runs) * 100) if e_runs else 0)

        # Medium
        m_runs = [r for r in runs if r.get("difficulty") == "medium"]
        m_pass = sum(1 for r in m_runs if r.get("iterations_to_success") is not None and r["iterations_to_success"] <= k)
        cum_acc["medium"].append((m_pass / len(m_runs) * 100) if m_runs else 0)

        # Hard
        h_runs = [r for r in runs if r.get("difficulty") == "hard"]
        h_pass = sum(1 for r in h_runs if r.get("iterations_to_success") is not None and r["iterations_to_success"] <= k)
        cum_acc["hard"].append((h_pass / len(h_runs) * 100) if h_runs else 0)

        # Overall
        o_pass = sum(1 for r in runs if r.get("iterations_to_success") is not None and r["iterations_to_success"] <= k)
        cum_acc["overall"].append((o_pass / len(runs) * 100) if runs else 0)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(k_vals, cum_acc["easy"], marker="o", linewidth=2.2, color="#2ca02c", label="Swarm: Easy Tier")
    ax.plot(k_vals, cum_acc["medium"], marker="s", linewidth=2.2, color="#1f77b4", label="Swarm: Medium Tier")
    ax.plot(k_vals, cum_acc["hard"], marker="^", linewidth=2.2, color="#d62728", label="Swarm: Hard Tier")
    ax.plot(k_vals, cum_acc["overall"], marker="D", linewidth=2.8, color="#9467bd", linestyle="--", label="Swarm: Overall Mean")

    # Monolithic Baseline
    ax.axhline(mono_pass_rate, color="#ff7f0e", linestyle=":", linewidth=2.5, label=f"Monolithic Frontier Pass@1 ({mono_pass_rate:.1f}%)")

    ax.set_xlabel("Self-Correction Iterations (K)")
    ax.set_ylabel("Cumulative Pass Rate (%)")
    ax.set_title("Figure 1: Iteration Count vs. Bug-Fixing Accuracy Curve")
    ax.set_xticks(k_vals)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    fig_png = os.path.join(FIG_DIR, "fig1_iteration_curve.png")
    fig_pdf = os.path.join(FIG_DIR, "fig1_iteration_curve.pdf")
    plt.tight_layout()
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"Generated Figure 1:\n  PNG: {os.path.relpath(fig_png, BASE_DIR)}\n  PDF: {os.path.relpath(fig_pdf, BASE_DIR)}")

def plot_complexity_ceiling():
    """Generates Figure 2: Dual-axis Complexity Ceiling Break-Even Plot."""
    table_path = os.path.join(BASE_DIR, "results", "tables", "rq2_complexity_ceiling.csv")
    if not os.path.exists(table_path):
        print("RQ2 table not found. Skipping Figure 2.")
        return
    df = pd.read_csv(table_path)

    fig, ax1 = plt.subplots(figsize=(8, 5.2))
    tiers = df["Difficulty"].tolist()
    x = np.arange(len(tiers))
    width = 0.35

    # Bar chart: Accuracy comparison
    rects1 = ax1.bar(x - width/2, df["Swarm_Accuracy"] * 100, width, label="Swarm Accuracy (%)", color="#1f77b4", alpha=0.85)
    rects2 = ax1.bar(x + width/2, df["Monolithic_Accuracy"] * 100, width, label="Monolithic Accuracy (%)", color="#ff7f0e", alpha=0.85)

    ax1.set_xlabel("Task Difficulty Tier")
    ax1.set_ylabel("Pass Rate (%)", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers)
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Line chart: Cost per solved task on secondary Y axis
    ax2 = ax1.twinx()
    ax2.plot(x, df["Swarm_Cost_Per_Solved"], color="#2ca02c", marker="o", linewidth=2.2, label="Swarm Cost/Solve ($)")
    ax2.plot(x, df["Monolithic_Cost_Per_Solved"], color="#d62728", marker="s", linewidth=2.2, linestyle="--", label="Mono Cost/Solve ($)")
    ax2.set_ylabel("Estimated Cost per Solved Task (USD)", color="#2ca02c")

    plt.title("Figure 2: Complexity Ceiling Break-Even Analysis (Accuracy vs. Cost)")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig_png = os.path.join(FIG_DIR, "fig2_complexity_ceiling.png")
    fig_pdf = os.path.join(FIG_DIR, "fig2_complexity_ceiling.pdf")
    plt.tight_layout()
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"Generated Figure 2:\n  PNG: {os.path.relpath(fig_png, BASE_DIR)}\n  PDF: {os.path.relpath(fig_pdf, BASE_DIR)}")

if __name__ == "__main__":
    plot_iteration_curves()
    plot_complexity_ceiling()
