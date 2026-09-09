"""
Comparative Analysis: OpenRouter (AkashML/MXFP4) vs. Groq (Native LPU).
Compares the two 90-call Prototype 1 baseline runs across:
- Overall Pass@1 accuracy & per-tier accuracy (Easy, Medium, Hard)
- Mean latency (seconds)
- Fallback usage rate (diff-resilience context-matching fallback)
- Token consumption (mean prompt, completion, total)
- Cost and execution throughput
"""
import os
import glob
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPENROUTER_DIR = os.path.join(BASE_DIR, "results", "prototype_run")
GROQ_DIR = os.path.join(BASE_DIR, "results", "prototype_run_groq")


def load_latest_csv(run_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(run_dir, "benchmark_prototype_*.csv"))
    if not files:
        files = glob.glob(os.path.join(run_dir, "benchmark_*.csv"))
    if not files:
        raise FileNotFoundError(f"No benchmark CSV found in {run_dir}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loaded: {os.path.relpath(latest, BASE_DIR)}")
    return pd.read_csv(latest)


def generate_comparison_table():
    df_or = load_latest_csv(OPENROUTER_DIR)
    df_groq = load_latest_csv(GROQ_DIR)

    def calc_metrics(df: pd.DataFrame, label: str, hw: str, quant: str):
        total_runs = len(df)
        acc_overall = df["final_pass"].mean() * 100
        easy_acc = df[df["difficulty"] == "easy"]["final_pass"].mean() * 100 if "easy" in df["difficulty"].values else 0.0
        med_acc = df[df["difficulty"] == "medium"]["final_pass"].mean() * 100 if "medium" in df["difficulty"].values else 0.0
        hard_acc = df[df["difficulty"] == "hard"]["final_pass"].mean() * 100 if "hard" in df["difficulty"].values else 0.0
        mean_lat = df["cumulative_wall_clock_sec"].mean()
        fallback_pct = (df["used_fallback"].mean() * 100) if "used_fallback" in df.columns else 0.0
        prompt_tok = df["total_prompt_tokens"].mean()
        comp_tok = df["total_completion_tokens"].mean()
        tot_tok = prompt_tok + comp_tok
        tot_cost = df["estimated_cost_usd"].sum()

        return {
            "Provider": label,
            "Hardware": hw,
            "Quantization / Serving": quant,
            "Total Runs": total_runs,
            "Pass@1 Overall (%)": round(acc_overall, 2),
            "Easy Pass@1 (%)": round(easy_acc, 2),
            "Medium Pass@1 (%)": round(med_acc, 2),
            "Hard Pass@1 (%)": round(hard_acc, 2),
            "Mean Latency (s)": round(mean_lat, 2),
            "Fallback Usage (%)": round(fallback_pct, 2),
            "Mean Prompt Tokens": round(prompt_tok, 1),
            "Mean Comp Tokens": round(comp_tok, 1),
            "Mean Total Tokens": round(tot_tok, 1),
            "Total Cost ($)": round(tot_cost, 5)
        }

    row_or = calc_metrics(df_or, "OpenRouter", "AkashML Cloud GPU", "MXFP4 (5.1B active)")
    row_groq = calc_metrics(df_groq, "Groq", "Groq LPU (Native)", "Direct LPU On-Demand")

    df_comp = pd.DataFrame([row_or, row_groq])

    # Save to GROQ_DIR
    out_csv = os.path.join(GROQ_DIR, "openrouter_vs_groq_comparison.csv")
    df_comp.to_csv(out_csv, index=False)
    print(f"\nSaved comparison CSV: {os.path.relpath(out_csv, BASE_DIR)}")

    # Format markdown table manually without tabulate dependency
    headers = list(df_comp.columns)
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df_comp.iterrows():
        md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md_table = "\n".join(md_lines)

    print("\n" + "=" * 80)
    print("COMPARISON: OpenRouter (AkashML/MXFP4) vs. Groq (Native LPU)")
    print("=" * 80)
    print(md_table)
    print("=" * 80 + "\n")

    return df_comp, md_table


if __name__ == "__main__":
    generate_comparison_table()
