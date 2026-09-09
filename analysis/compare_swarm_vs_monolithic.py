"""
Comparative Statistical Analysis: Multi-SLM Swarm vs. Monolithic Groq LPU Baseline.
Phase 2 Comparison:
- Overall & Per-tier Pass@1 Accuracy (Easy, Medium, Hard)
- Latency (cumulative wall-clock seconds)
- Token consumption (prompt, completion, total)
- Fallback usage rate (context-matching fallback in sandbox)
- QA exhausted rate (Swarm retry-exhaustion forced through)
- Statistical significance tests:
    * Independent two-sample t-test (Welch's t-test)
    * Cohen's d effect size per tier
Exports to: results/prototype_run_swarm/swarm_vs_groq_comparison.csv and prints markdown table.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GROQ_DIR = os.path.join(BASE_DIR, "results", "prototype_run_groq")
SWARM_DIR = os.path.join(BASE_DIR, "results", "prototype_run_swarm")


def load_latest_csv(run_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(run_dir, "benchmark_prototype_*.csv"))
    if not files:
        files = glob.glob(os.path.join(run_dir, "benchmark_*.csv"))
    if not files:
        raise FileNotFoundError(f"No benchmark CSV found in {run_dir}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loaded: {os.path.relpath(latest, BASE_DIR)}")
    return pd.read_csv(latest)


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between two independent samples."""
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    # Pooled standard deviation
    s_pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if s_pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / s_pooled)


def run_comparison():
    df_groq = load_latest_csv(GROQ_DIR)
    df_swarm = load_latest_csv(SWARM_DIR)

    tiers = ["overall", "easy", "medium", "hard"]
    records = []

    for tier in tiers:
        if tier == "overall":
            sub_groq = df_groq
            sub_swarm = df_swarm
        else:
            sub_groq = df_groq[df_groq["difficulty"] == tier]
            sub_swarm = df_swarm[df_swarm["difficulty"] == tier]

        n_groq = len(sub_groq)
        n_swarm = len(sub_swarm)

        # Pass@1
        acc_groq = sub_groq["final_pass"].mean() * 100.0 if n_groq else 0.0
        acc_swarm = sub_swarm["final_pass"].mean() * 100.0 if n_swarm else 0.0
        delta_acc = acc_swarm - acc_groq

        # Latency
        lat_groq = sub_groq["cumulative_wall_clock_sec"].mean() if n_groq else 0.0
        lat_swarm = sub_swarm["cumulative_wall_clock_sec"].mean() if n_swarm else 0.0
        delta_lat = lat_swarm - lat_groq

        # Prompt & Completion tokens
        p_tok_groq = sub_groq["total_prompt_tokens"].mean() if n_groq else 0.0
        p_tok_swarm = sub_swarm["total_prompt_tokens"].mean() if n_swarm else 0.0
        c_tok_groq = sub_groq["total_completion_tokens"].mean() if n_groq else 0.0
        c_tok_swarm = sub_swarm["total_completion_tokens"].mean() if n_swarm else 0.0
        tot_tok_groq = p_tok_groq + c_tok_groq
        tot_tok_swarm = p_tok_swarm + c_tok_swarm

        # Fallback rate
        fb_groq = (sub_groq["used_fallback"].mean() * 100.0) if ("used_fallback" in sub_groq.columns and n_groq) else 0.0
        fb_swarm = (sub_swarm["used_fallback"].mean() * 100.0) if ("used_fallback" in sub_swarm.columns and n_swarm) else 0.0

        # QA Exhausted rate
        qa_ex_swarm = (sub_swarm["qa_exhausted"].mean() * 100.0) if ("qa_exhausted" in sub_swarm.columns and n_swarm) else 0.0

        # Statistical tests: Pass@1 Welch's t-test
        if n_groq > 1 and n_swarm > 1:
            t_stat_acc, p_val_acc = stats.ttest_ind(
                sub_swarm["final_pass"].astype(float),
                sub_groq["final_pass"].astype(float),
                equal_var=False
            )
            d_acc = compute_cohens_d(
                sub_swarm["final_pass"].astype(float).values,
                sub_groq["final_pass"].astype(float).values
            )

            t_stat_lat, p_val_lat = stats.ttest_ind(
                sub_swarm["cumulative_wall_clock_sec"].astype(float),
                sub_groq["cumulative_wall_clock_sec"].astype(float),
                equal_var=False
            )
            d_lat = compute_cohens_d(
                sub_swarm["cumulative_wall_clock_sec"].astype(float).values,
                sub_groq["cumulative_wall_clock_sec"].astype(float).values
            )
        else:
            t_stat_acc, p_val_acc, d_acc = 0.0, 1.0, 0.0
            t_stat_lat, p_val_lat, d_lat = 0.0, 1.0, 0.0

        records.append({
            "Tier": tier.capitalize(),
            "Groq N": n_groq,
            "Swarm N": n_swarm,
            "Groq Pass@1 (%)": round(acc_groq, 2),
            "Swarm Pass@1 (%)": round(acc_swarm, 2),
            "Delta Pass@1 (% pts)": round(delta_acc, 2),
            "t-stat (Acc)": round(t_stat_acc, 3),
            "p-val (Acc)": round(p_val_acc, 4),
            "Cohen's d (Acc)": round(d_acc, 3),
            "Groq Mean Lat (s)": round(lat_groq, 2),
            "Swarm Mean Lat (s)": round(lat_swarm, 2),
            "Delta Lat (s)": round(delta_lat, 2),
            "t-stat (Lat)": round(t_stat_lat, 3),
            "p-val (Lat)": round(p_val_lat, 4),
            "Cohen's d (Lat)": round(d_lat, 3),
            "Groq Mean Tokens": round(tot_tok_groq, 1),
            "Swarm Mean Tokens": round(tot_tok_swarm, 1),
            "Groq Fallback (%)": round(fb_groq, 2),
            "Swarm Fallback (%)": round(fb_swarm, 2),
            "Swarm QA Exhausted (%)": round(qa_ex_swarm, 2)
        })

    df_comp = pd.DataFrame(records)

    # Export to SWARM_DIR
    os.makedirs(SWARM_DIR, exist_ok=True)
    out_csv = os.path.join(SWARM_DIR, "swarm_vs_groq_comparison.csv")
    df_comp.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSaved comparison CSV to: {os.path.relpath(out_csv, BASE_DIR)}")

    # Format Markdown Table
    headers = list(df_comp.columns)
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df_comp.iterrows():
        md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md_table = "\n".join(md_lines)

    print("\n" + "=" * 100)
    print("PHASE 2 COMPARISON: Multi-SLM Swarm vs. Monolithic Groq LPU Baseline")
    print("=" * 100)
    print(md_table)
    print("=" * 100 + "\n")

    # Complexity ceiling analysis
    print("COMPLEXITY CEILING ANALYSIS:")
    for row in records:
        t = row["Tier"]
        d_val = row["Cohen's d (Acc)"]
        p_val = row["p-val (Acc)"]
        delta = row["Delta Pass@1 (% pts)"]
        sig = "statistically significant" if p_val < 0.05 else "not statistically significant"
        print(f"  * {t} Tier: Swarm vs Groq difference = {delta:+0.2f}% (d = {d_val}, p = {p_val}, {sig})")

    return df_comp, md_table


if __name__ == "__main__":
    run_comparison()
