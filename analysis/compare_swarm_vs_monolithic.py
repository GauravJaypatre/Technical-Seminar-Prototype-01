"""
Comparative Statistical Analysis:
Three-Way Benchmark Comparison:
1. Groq Native LPU Baseline (openai/gpt-oss-120b)
2. Multi-SLM Swarm v1 (Header-Only QA)
3. Multi-SLM Swarm v2 (Content-Validated QA)

Metrics Evaluated:
- Pass@1 Accuracy (% per tier and overall)
- Wall-clock Latency (mean seconds)
- Token Consumption (prompt, completion, total)
- Patch Applied Rate (% successfully applied in sandbox)
- Context-Matching Fallback Rate (%)
- QA Exhaustion Rate (%) & Mean QA Retries
- Two-sample Welch's t-test and Cohen's d per tier

Exports:
- results/prototype_run_swarm_v2_content_validated_qa/three_way_comparison_groq_v1_v2.csv
- results/prototype_run_swarm_v2_content_validated_qa/swarm_v2_vs_groq_comparison.csv
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GROQ_DIR = os.path.join(BASE_DIR, "results", "prototype_run_groq")
SWARM_V1_DIR = os.path.join(BASE_DIR, "results", "prototype_run_swarm_v1_header_only_qa")
SWARM_V2_DIR = os.path.join(BASE_DIR, "results", "prototype_run_swarm_v2_content_validated_qa")


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
    s_pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if s_pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / s_pooled)


def run_comparison(v2_dir: str = None):
    target_v2_dir = v2_dir or SWARM_V2_DIR
    df_groq = load_latest_csv(GROQ_DIR)
    df_v1 = load_latest_csv(SWARM_V1_DIR)
    df_v2 = load_latest_csv(target_v2_dir)

    tiers = ["overall", "easy", "medium", "hard"]

    # 1. Three-Way High-Level Summary Table
    three_way_records = []
    for system_label, df_sys in [("Groq Monolithic (120B)", df_groq),
                                  ("Swarm v1 (Header-Only QA)", df_v1),
                                  ("Swarm v2 (Content-Validated QA)", df_v2)]:
        for tier in tiers:
            if tier == "overall":
                sub = df_sys
            else:
                sub = df_sys[df_sys["difficulty"] == tier]
            n = len(sub)
            acc = sub["final_pass"].mean() * 100.0 if n else 0.0
            lat = sub["cumulative_wall_clock_sec"].mean() if n else 0.0
            p_tok = sub["total_prompt_tokens"].mean() if n else 0.0
            c_tok = sub["total_completion_tokens"].mean() if n else 0.0
            fb = (sub["used_fallback"].mean() * 100.0) if ("used_fallback" in sub.columns and n) else 0.0
            qa_ex = (sub["qa_exhausted"].mean() * 100.0) if ("qa_exhausted" in sub.columns and n) else 0.0
            qa_ret = sub["qa_retries"].mean() if ("qa_retries" in sub.columns and n) else 0.0

            three_way_records.append({
                "System": system_label,
                "Tier": tier.capitalize(),
                "N": n,
                "Pass@1 (%)": round(acc, 2),
                "Mean Latency (s)": round(lat, 2),
                "Mean Prompt Tokens": round(p_tok, 1),
                "Mean Comp Tokens": round(c_tok, 1),
                "Total Tokens": round(p_tok + c_tok, 1),
                "Fallback Rate (%)": round(fb, 2),
                "QA Exhausted (%)": round(qa_ex, 2),
                "Mean QA Retries": round(qa_ret, 2)
            })

    df_three_way = pd.DataFrame(three_way_records)

    # 2. Pairwise Comparison: Swarm v2 vs. Groq
    comp_records = []
    for tier in tiers:
        if tier == "overall":
            sub_groq = df_groq
            sub_v2 = df_v2
        else:
            sub_groq = df_groq[df_groq["difficulty"] == tier]
            sub_v2 = df_v2[df_v2["difficulty"] == tier]

        n_groq = len(sub_groq)
        n_v2 = len(sub_v2)

        acc_groq = sub_groq["final_pass"].mean() * 100.0 if n_groq else 0.0
        acc_v2 = sub_v2["final_pass"].mean() * 100.0 if n_v2 else 0.0
        delta_acc = acc_v2 - acc_groq

        lat_groq = sub_groq["cumulative_wall_clock_sec"].mean() if n_groq else 0.0
        lat_v2 = sub_v2["cumulative_wall_clock_sec"].mean() if n_v2 else 0.0
        delta_lat = lat_v2 - lat_groq

        p_tok_groq = sub_groq["total_prompt_tokens"].mean() if n_groq else 0.0
        p_tok_v2 = sub_v2["total_prompt_tokens"].mean() if n_v2 else 0.0
        c_tok_groq = sub_groq["total_completion_tokens"].mean() if n_groq else 0.0
        c_tok_v2 = sub_v2["total_completion_tokens"].mean() if n_v2 else 0.0

        fb_groq = (sub_groq["used_fallback"].mean() * 100.0) if ("used_fallback" in sub_groq.columns and n_groq) else 0.0
        fb_v2 = (sub_v2["used_fallback"].mean() * 100.0) if ("used_fallback" in sub_v2.columns and n_v2) else 0.0

        qa_ex_v2 = (sub_v2["qa_exhausted"].mean() * 100.0) if ("qa_exhausted" in sub_v2.columns and n_v2) else 0.0

        if n_groq > 1 and n_v2 > 1:
            t_stat_acc, p_val_acc = stats.ttest_ind(
                sub_v2["final_pass"].astype(float),
                sub_groq["final_pass"].astype(float),
                equal_var=False
            )
            d_acc = compute_cohens_d(
                sub_v2["final_pass"].astype(float).values,
                sub_groq["final_pass"].astype(float).values
            )
            t_stat_lat, p_val_lat = stats.ttest_ind(
                sub_v2["cumulative_wall_clock_sec"].astype(float),
                sub_groq["cumulative_wall_clock_sec"].astype(float),
                equal_var=False
            )
            d_lat = compute_cohens_d(
                sub_v2["cumulative_wall_clock_sec"].astype(float).values,
                sub_groq["cumulative_wall_clock_sec"].astype(float).values
            )
        else:
            t_stat_acc, p_val_acc, d_acc = 0.0, 1.0, 0.0
            t_stat_lat, p_val_lat, d_lat = 0.0, 1.0, 0.0

        comp_records.append({
            "Tier": tier.capitalize(),
            "Groq N": n_groq,
            "Swarm v2 N": n_v2,
            "Groq Pass@1 (%)": round(acc_groq, 2),
            "Swarm v2 Pass@1 (%)": round(acc_v2, 2),
            "Delta Pass@1 (% pts)": round(delta_acc, 2),
            "t-stat (Acc)": round(t_stat_acc, 3),
            "p-val (Acc)": round(p_val_acc, 4),
            "Cohen's d (Acc)": round(d_acc, 3),
            "Groq Mean Lat (s)": round(lat_groq, 2),
            "Swarm v2 Mean Lat (s)": round(lat_v2, 2),
            "Delta Lat (s)": round(delta_lat, 2),
            "t-stat (Lat)": round(t_stat_lat, 3),
            "p-val (Lat)": round(p_val_lat, 4),
            "Cohen's d (Lat)": round(d_lat, 3),
            "Groq Mean Tokens": round(p_tok_groq + c_tok_groq, 1),
            "Swarm v2 Mean Tokens": round(p_tok_v2 + c_tok_v2, 1),
            "Groq Fallback (%)": round(fb_groq, 2),
            "Swarm v2 Fallback (%)": round(fb_v2, 2),
            "Swarm v2 QA Exhausted (%)": round(qa_ex_v2, 2)
        })

    df_comp = pd.DataFrame(comp_records)

    # Save CSVs to target_v2_dir
    os.makedirs(target_v2_dir, exist_ok=True)
    out_3way_csv = os.path.join(target_v2_dir, "three_way_comparison_groq_v1_v2.csv")
    df_three_way.to_csv(out_3way_csv, index=False, encoding="utf-8")
    print(f"\nSaved 3-Way Comparison CSV to: {os.path.relpath(out_3way_csv, BASE_DIR)}")

    out_comp_csv = os.path.join(target_v2_dir, "swarm_v2_vs_groq_comparison.csv")
    df_comp.to_csv(out_comp_csv, index=False, encoding="utf-8")
    print(f"Saved Swarm v2 vs Groq CSV to: {os.path.relpath(out_comp_csv, BASE_DIR)}")

    # Format Markdown Table for Three-Way
    headers_3way = list(df_three_way.columns)
    md_lines_3way = []
    md_lines_3way.append("| " + " | ".join(headers_3way) + " |")
    md_lines_3way.append("| " + " | ".join(["---"] * len(headers_3way)) + " |")
    for _, row in df_three_way.iterrows():
        md_lines_3way.append("| " + " | ".join(str(row[h]) for h in headers_3way) + " |")
    md_table_3way = "\n".join(md_lines_3way)

    print("\n" + "=" * 100)
    print("THREE-WAY BENCHMARK COMPARISON: Groq LPU vs. Swarm v1 vs. Swarm v2")
    print("=" * 100)
    print(md_table_3way)
    print("=" * 100 + "\n")

    return df_three_way, df_comp, md_table_3way


if __name__ == "__main__":
    run_comparison()
