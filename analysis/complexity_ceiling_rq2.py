"""
Statistical Analysis for RQ2:
Complexity Ceiling and Break-Even Point.
At what difficulty level does the swarm's cumulative cost/latency exceed
the monolithic baseline's cost for equivalent accuracy?
"""
import os
import sys
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_rq2_analysis(swarm_csv: str = None, mono_csv: str = None):
    # Find latest swarm and monolithic CSVs if not passed
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

    print(f"Analyzing Complexity Ceiling (RQ2):")
    print(f"  Swarm Data:      {os.path.relpath(swarm_csv, BASE_DIR)}")
    print(f"  Monolithic Data: {os.path.relpath(mono_csv, BASE_DIR)}")

    df_swarm = pd.read_csv(swarm_csv)
    df_mono = pd.read_csv(mono_csv)

    tiers = ["easy", "medium", "hard"]
    results = []

    for tier in tiers:
        s_tier = df_swarm[df_swarm["difficulty"] == tier]
        m_tier = df_mono[df_mono["difficulty"] == tier]

        # Accuracy
        s_acc = s_tier["final_pass"].mean() if len(s_tier) > 0 else 0.0
        m_acc = m_tier["final_pass"].mean() if len(m_tier) > 0 else 0.0
        delta_acc = s_acc - m_acc

        # Cost
        s_cost_total = s_tier["estimated_cost_usd"].sum()
        m_cost_total = m_tier["estimated_cost_usd"].sum()
        s_solved = s_tier["final_pass"].sum()
        m_solved = m_tier["final_pass"].sum()

        s_cost_per_solve = (s_cost_total / s_solved) if s_solved > 0 else float("inf")
        m_cost_per_solve = (m_cost_total / m_solved) if m_solved > 0 else float("inf")
        cost_ratio = (s_cost_per_solve / m_cost_per_solve) if m_cost_per_solve > 0 else float("inf")

        # Latency
        s_lat_mean = s_tier["cumulative_wall_clock_sec"].mean()
        m_lat_mean = m_tier["cumulative_wall_clock_sec"].mean()
        lat_ratio = (s_lat_mean / m_lat_mean) if m_lat_mean > 0 else float("inf")

        # Ceiling check: swarm fails to match accuracy (delta_acc < 0) or cost exceeds monolithic
        is_ceiling = (delta_acc <= 0) or (cost_ratio > 1.0)

        results.append({
            "Difficulty": tier.capitalize(),
            "Swarm_Accuracy": round(s_acc, 3),
            "Monolithic_Accuracy": round(m_acc, 3),
            "Accuracy_Delta": round(delta_acc, 3),
            "Swarm_Cost_Per_Solved": round(s_cost_per_solve, 5),
            "Monolithic_Cost_Per_Solved": round(m_cost_per_solve, 5),
            "Cost_Ratio_Swarm_vs_Mono": round(cost_ratio, 3),
            "Swarm_Mean_Latency_Sec": round(s_lat_mean, 2),
            "Monolithic_Mean_Latency_Sec": round(m_lat_mean, 2),
            "Latency_Ratio": round(lat_ratio, 2),
            "Exceeds_Complexity_Ceiling": is_ceiling
        })

    out_df = pd.DataFrame(results)
    print("\n--- RQ2 Complexity Ceiling & Break-Even Matrix ---")
    print(out_df.to_string(index=False))

    # Identify exact ceiling tier
    ceiling_tiers = [r["Difficulty"] for r in results if r["Exceeds_Complexity_Ceiling"]]
    print("\nEmpirical Findings:")
    if ceiling_tiers:
        print(f"  Identified Complexity Ceiling at tier: '{ceiling_tiers[0]}'")
        print(f"  At this tier, the swarm either ceases to match frontier accuracy or its cumulative iterations render it economically inferior.")
    else:
        print("  Swarm maintains accuracy advantage or parity across all tested tiers.")

    out_path = os.path.join(BASE_DIR, "results", "tables", "rq2_complexity_ceiling.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nComplexity ceiling table saved to: {os.path.relpath(out_path, BASE_DIR)}")
    return out_df

if __name__ == "__main__":
    s_path = sys.argv[1] if len(sys.argv) > 1 else None
    m_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_rq2_analysis(s_path, m_path)
