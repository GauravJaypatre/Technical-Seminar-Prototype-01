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
import statsmodels.stats.proportion as sm_prop

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
    """Compute Cohen's d effect size between two independent samples (continuous metrics only)."""
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

    # 1. Three-Way High-Level Summary Table with Wilson Score CIs
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
            k = int(sub["final_pass"].sum()) if n else 0
            acc = (k / n) * 100.0 if n else 0.0
            
            # Wilson score confidence interval
            if n > 0:
                ci_l, ci_u = sm_prop.proportion_confint(k, n, alpha=0.05, method="wilson")
                ci_l_pct, ci_u_pct = ci_l * 100.0, ci_u * 100.0
            else:
                ci_l_pct, ci_u_pct = 0.0, 0.0

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
                "Pass Count": k,
                "Pass@1 (%)": round(acc, 2),
                "Wilson 95% CI Low (%)": round(ci_l_pct, 2),
                "Wilson 95% CI High (%)": round(ci_u_pct, 2),
                "Pass@1 [95% CI]": f"{acc:.2f}% [{ci_l_pct:.2f}%, {ci_u_pct:.2f}%]",
                "Mean Latency (s)": round(lat, 2),
                "Mean Prompt Tokens": round(p_tok, 1),
                "Mean Comp Tokens": round(c_tok, 1),
                "Total Tokens": round(p_tok + c_tok, 1),
                "Fallback Rate (%)": round(fb, 2),
                "QA Exhausted (%)": round(qa_ex, 2),
                "Mean QA Retries": round(qa_ret, 2)
            })

    df_three_way = pd.DataFrame(three_way_records)

    # 2. Pairwise Comparison: Swarm v2 vs. Groq with Proportion-Appropriate Inference
    comp_v1_records = []
    comp_v2_stats_records = []

    for tier in tiers:
        if tier == "overall":
            sub_groq = df_groq
            sub_v2 = df_v2
        else:
            sub_groq = df_groq[df_groq["difficulty"] == tier]
            sub_v2 = df_v2[df_v2["difficulty"] == tier]

        n_groq = len(sub_groq)
        n_v2 = len(sub_v2)
        k_groq = int(sub_groq["final_pass"].sum()) if n_groq else 0
        k_v2 = int(sub_v2["final_pass"].sum()) if n_v2 else 0

        acc_groq = (k_groq / n_groq) * 100.0 if n_groq else 0.0
        acc_v2 = (k_v2 / n_v2) * 100.0 if n_v2 else 0.0
        delta_acc = acc_v2 - acc_groq

        # Wilson CIs
        ci_l_g, ci_u_g = sm_prop.proportion_confint(k_groq, n_groq, alpha=0.05, method="wilson")
        ci_l_v2, ci_u_v2 = sm_prop.proportion_confint(k_v2, n_v2, alpha=0.05, method="wilson")

        # Fisher's exact test (primary for small N and zero-count cells)
        table = [[k_v2, n_v2 - k_v2], [k_groq, n_groq - k_groq]]
        odds_ratio, p_fisher = stats.fisher_exact(table)

        # Haldane-Anscombe correction (+0.5 added to all 4 cells) to resolve zero-count degeneracy
        a, b = k_v2, n_v2 - k_v2
        c, d = k_groq, n_groq - k_groq
        ha_odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        effective_odds_ratio = ha_odds_ratio if (k_v2 == 0 or k_groq == 0) else odds_ratio

        # Two-proportion z-test (secondary check)
        try:
            z_stat, p_ztest = sm_prop.proportions_ztest([k_v2, k_groq], [n_v2, n_groq])
        except Exception:
            z_stat, p_ztest = np.nan, np.nan

        # Risk difference 95% CI (Newcombe hybrid score method)
        ci_l_rd, ci_u_rd = sm_prop.confint_proportions_2indep(k_v2, n_v2, k_groq, n_groq, method="newcomb")

        # Latency statistics (continuous outcome -> Welch's t-test and Cohen's d appropriate)
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
            d_acc_continuous_proxy = compute_cohens_d(
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
            t_stat_acc, p_val_acc, d_acc_continuous_proxy = 0.0, 1.0, 0.0
            t_stat_lat, p_val_lat, d_lat = 0.0, 1.0, 0.0

        # Legacy comparison record (preserved)
        comp_v1_records.append({
            "Tier": tier.capitalize(),
            "Groq N": n_groq,
            "Swarm v2 N": n_v2,
            "Groq Pass@1 (%)": round(acc_groq, 2),
            "Swarm v2 Pass@1 (%)": round(acc_v2, 2),
            "Delta Pass@1 (% pts)": round(delta_acc, 2),
            "t-stat (Acc)": round(t_stat_acc, 3),
            "p-val (Acc)": round(p_val_acc, 4),
            "Cohen's d (Acc)": round(d_acc_continuous_proxy, 3),
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

        # Corrected Proportion-Appropriate Inference Record (v2 stats)
        comp_v2_stats_records.append({
            "Tier": tier.capitalize(),
            "Groq N": n_groq,
            "Swarm v2 N": n_v2,
            "Groq Pass Count": k_groq,
            "Swarm v2 Pass Count": k_v2,
            "Groq Pass@1 (%)": round(acc_groq, 2),
            "Groq Wilson CI Low (%)": round(ci_l_g * 100.0, 2),
            "Groq Wilson CI High (%)": round(ci_u_g * 100.0, 2),
            "Swarm v2 Pass@1 (%)": round(acc_v2, 2),
            "Swarm v2 Wilson CI Low (%)": round(ci_l_v2 * 100.0, 2),
            "Swarm v2 Wilson CI High (%)": round(ci_u_v2 * 100.0, 2),
            "Risk Difference (% pts)": round(delta_acc, 2),
            "Risk Diff 95% CI Low (%)": round(ci_l_rd * 100.0, 2),
            "Risk Diff 95% CI High (%)": round(ci_u_rd * 100.0, 2),
            "Fisher Odds Ratio (Raw)": round(odds_ratio, 4),
            "Fisher Odds Ratio (Haldane-Anscombe)": round(ha_odds_ratio, 4),
            "Reported Odds Ratio": round(effective_odds_ratio, 4),
            "Fisher p-val (Acc)": float(f"{p_fisher:.4e}"),
            "Two-Prop z-stat": round(z_stat, 3),
            "Two-Prop p-val (Acc)": float(f"{p_ztest:.4e}"),
            "Welch t-stat (Acc, linear-proxy)": round(t_stat_acc, 3),
            "Welch p-val (Acc, linear-proxy)": float(f"{p_val_acc:.4e}"),
            "Groq Mean Lat (s)": round(lat_groq, 2),
            "Swarm v2 Mean Lat (s)": round(lat_v2, 2),
            "Delta Lat (s)": round(delta_lat, 2),
            "Welch t-stat (Lat)": round(t_stat_lat, 3),
            "Welch p-val (Lat)": float(f"{p_val_lat:.4e}"),
            "Cohen's d (Lat)": round(d_lat, 3),
            "Groq Mean Tokens": round(p_tok_groq + c_tok_groq, 1),
            "Swarm v2 Mean Tokens": round(p_tok_v2 + c_tok_v2, 1),
            "Groq Fallback (%)": round(fb_groq, 2),
            "Swarm v2 Fallback (%)": round(fb_v2, 2),
            "Swarm v2 QA Exhausted (%)": round(qa_ex_v2, 2)
        })

    df_comp_v1 = pd.DataFrame(comp_v1_records)
    df_comp_v2_stats = pd.DataFrame(comp_v2_stats_records)

    # Save CSVs to target_v2_dir
    os.makedirs(target_v2_dir, exist_ok=True)
    out_3way_csv = os.path.join(target_v2_dir, "three_way_comparison_groq_v1_v2.csv")
    df_three_way.to_csv(out_3way_csv, index=False, encoding="utf-8")
    print(f"\nSaved 3-Way Comparison CSV to: {os.path.relpath(out_3way_csv, BASE_DIR)}")

    # Preserve original comparison CSV
    out_comp_csv = os.path.join(target_v2_dir, "swarm_v2_vs_groq_comparison.csv")
    df_comp_v1.to_csv(out_comp_csv, index=False, encoding="utf-8")
    print(f"Preserved original Swarm v2 vs Groq CSV: {os.path.relpath(out_comp_csv, BASE_DIR)}")

    # Export new proportion-appropriate stats CSV
    out_comp_v2_stats_csv = os.path.join(target_v2_dir, "swarm_v2_vs_groq_comparison_v2_stats.csv")
    df_comp_v2_stats.to_csv(out_comp_v2_stats_csv, index=False, encoding="utf-8")
    print(f"Saved Corrected V2 Stats CSV to: {os.path.relpath(out_comp_v2_stats_csv, BASE_DIR)}")

    # Format Markdown Table for Three-Way
    display_3way_cols = [
        "System", "Tier", "N", "Pass Count", "Pass@1 [95% CI]",
        "Mean Latency (s)", "Total Tokens", "Fallback Rate (%)",
        "QA Exhausted (%)", "Mean QA Retries"
    ]
    md_lines_3way = [
        "| " + " | ".join(display_3way_cols) + " |",
        "| " + " | ".join(["---"] * len(display_3way_cols)) + " |"
    ]
    for _, row in df_three_way.iterrows():
        md_lines_3way.append("| " + " | ".join(str(row[h]) for h in display_3way_cols) + " |")
    md_table_3way = "\n".join(md_lines_3way)

    # Format Markdown Table for Corrected Pairwise Stats
    display_comp_cols = [
        "Tier", "Groq N", "Swarm v2 N", "Groq Pass@1 (%)", "Swarm v2 Pass@1 (%)",
        "Risk Difference (% pts)", "Fisher OR (Raw)", "Fisher OR (Haldane-Anscombe)",
        "Fisher p-val (Acc)", "Two-Prop z-stat", "Two-Prop p-val (Acc)", "Delta Lat (s)", "Cohen's d (Lat)"
    ]
    md_lines_comp = [
        "| " + " | ".join(display_comp_cols) + " |",
        "| " + " | ".join(["---"] * len(display_comp_cols)) + " |"
    ]
    for _, row in df_comp_v2_stats.iterrows():
        d_lat_val = row["Cohen's d (Lat)"]
        row_disp = {
            "Tier": row["Tier"],
            "Groq N": row["Groq N"],
            "Swarm v2 N": row["Swarm v2 N"],
            "Groq Pass@1 (%)": f"{row['Groq Pass@1 (%)']:.2f}% [{row['Groq Wilson CI Low (%)']:.2f}%, {row['Groq Wilson CI High (%)']:.2f}%]",
            "Swarm v2 Pass@1 (%)": f"{row['Swarm v2 Pass@1 (%)']:.2f}% [{row['Swarm v2 Wilson CI Low (%)']:.2f}%, {row['Swarm v2 Wilson CI High (%)']:.2f}%]",
            "Risk Difference (% pts)": f"{row['Risk Difference (% pts)']:.2f}% [{row['Risk Diff 95% CI Low (%)']:.2f}%, {row['Risk Diff 95% CI High (%)']:.2f}%]",
            "Fisher OR (Raw)": f"{row['Fisher Odds Ratio (Raw)']:.4f}",
            "Fisher OR (Haldane-Anscombe)": f"{row['Fisher Odds Ratio (Haldane-Anscombe)']:.4f}",
            "Fisher p-val (Acc)": f"{row['Fisher p-val (Acc)']:.4e}",
            "Two-Prop z-stat": f"{row['Two-Prop z-stat']:.3f}",
            "Two-Prop p-val (Acc)": f"{row['Two-Prop p-val (Acc)']:.4e}",
            "Delta Lat (s)": f"{row['Delta Lat (s)']:.2f}s",
            "Cohen's d (Lat)": f"{d_lat_val:.3f}"
        }
        md_lines_comp.append("| " + " | ".join(str(row_disp[h]) for h in display_comp_cols) + " |")
    md_lines_comp.append("\n*Note: Haldane-Anscombe correction (+0.5 added to all four 2x2 cells) resolves zero-count cell degeneracy on Medium (0/30 vs. 27/30 -> OR=0.0021) and Hard (0/30 vs. 28/30 -> OR=0.0014).*")
    md_table_comp = "\n".join(md_lines_comp)

    print("\n" + "=" * 100)
    print("THREE-WAY BENCHMARK COMPARISON: Groq LPU vs. Swarm v1 vs. Swarm v2 (with Wilson 95% CIs)")
    print("=" * 100)
    print(md_table_3way)
    print("=" * 100 + "\n")

    print("\n" + "=" * 100)
    print("CORRECTED STATISTICAL COMPARISON: Swarm v2 vs. Groq (Proportion-Appropriate Methods)")
    print("=" * 100)
    print(md_table_comp)
    print("=" * 100 + "\n")

    return df_three_way, df_comp_v2_stats, md_table_3way, md_table_comp


if __name__ == "__main__":
    run_comparison()
