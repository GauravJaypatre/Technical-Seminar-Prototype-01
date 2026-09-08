"""
Generates master results tables matching the reference repository's Results.xlsx format:
- Sheets: 'RQ1_Iterations', 'RQ2_Complexity_Ceiling', 'RQ3_Accuracy', 'Per_Task_Breakdown', 'Router_Ablation'
Also exports clean markdown and CSV tables for seminar presentation.
"""
import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

def generate_master_results():
    # 1. Locate data files
    swarm_files = [f for f in glob.glob(os.path.join(BASE_DIR, "results", "runs", "benchmark_swarm_*.csv")) if "ablate" not in f and os.path.getsize(f) > 1000]
    mono_files = [f for f in glob.glob(os.path.join(BASE_DIR, "results", "runs", "benchmark_monolithic_*.csv")) if os.path.getsize(f) > 1000]
    ablate_files = [f for f in glob.glob(os.path.join(BASE_DIR, "results", "runs", "*ablate_easy*.csv")) if os.path.getsize(f) > 1000]

    if not swarm_files or not mono_files:
        print("Missing required run CSV files.")
        return

    swarm_csv = max(swarm_files, key=os.path.getmtime)
    mono_csv = max(mono_files, key=os.path.getmtime)
    ablate_csv = max(ablate_files, key=os.path.getmtime) if ablate_files else None

    df_swarm = pd.read_csv(swarm_csv)
    df_mono = pd.read_csv(mono_csv)

    # -------------------------------------------------------------
    # Sheet 1: RQ1 Iterations Summary
    # -------------------------------------------------------------
    rq1_csv = os.path.join(TABLES_DIR, "rq1_anova_summary.csv")
    df_rq1 = pd.read_csv(rq1_csv) if os.path.exists(rq1_csv) else pd.DataFrame()

    # -------------------------------------------------------------
    # Sheet 2: RQ2 Complexity Ceiling Summary
    # -------------------------------------------------------------
    rq2_csv = os.path.join(TABLES_DIR, "rq2_complexity_ceiling.csv")
    df_rq2 = pd.read_csv(rq2_csv) if os.path.exists(rq2_csv) else pd.DataFrame()

    # -------------------------------------------------------------
    # Sheet 3: RQ3 Accuracy Decomposition
    # -------------------------------------------------------------
    rq3_csv = os.path.join(TABLES_DIR, "rq3_hypothesis_test.csv")
    df_rq3 = pd.read_csv(rq3_csv) if os.path.exists(rq3_csv) else pd.DataFrame()

    # -------------------------------------------------------------
    # Sheet 4: Per-Task Granular Breakdown (All 18 tasks)
    # -------------------------------------------------------------
    s_task = df_swarm.groupby(["task_id", "difficulty"]).agg(
        Swarm_Pass_at_1=("pass_at_1", "mean"),
        Swarm_Final_Pass=("final_pass", "mean"),
        Swarm_Mean_Iterations=("iterations_to_success", lambda s: pd.to_numeric(s, errors="coerce").fillna(6).mean()),
        Swarm_Mean_Latency_Sec=("cumulative_wall_clock_sec", "mean"),
        Swarm_Mean_Tokens=("total_completion_tokens", "mean")
    ).reset_index()

    m_task = df_mono.groupby(["task_id", "difficulty"]).agg(
        Monolithic_Pass_at_1=("final_pass", "mean"),
        Monolithic_Mean_Latency_Sec=("cumulative_wall_clock_sec", "mean"),
        Monolithic_Mean_Tokens=("total_completion_tokens", "mean"),
        Monolithic_Cost_USD=("estimated_cost_usd", "mean")
    ).reset_index()

    per_task = pd.merge(s_task, m_task, on=["task_id", "difficulty"])
    per_task["Self_Correction_Gain"] = per_task["Swarm_Final_Pass"] - per_task["Swarm_Pass_at_1"]
    per_task = per_task.round(3)

    # Export Per-Task CSV
    per_task_path = os.path.join(TABLES_DIR, "per_task_results.csv")
    per_task.to_csv(per_task_path, index=False)

    # -------------------------------------------------------------
    # Sheet 5: Router Ablation Summary
    # -------------------------------------------------------------
    ablate_summary_csv = os.path.join(TABLES_DIR, "router_ablation_summary.csv")
    df_ablate = pd.read_csv(ablate_summary_csv) if os.path.exists(ablate_summary_csv) else pd.DataFrame()

    # -------------------------------------------------------------
    # Write to Multi-Sheet Excel: Results.xlsx
    # -------------------------------------------------------------
    excel_path = os.path.join(TABLES_DIR, "Results.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            if not df_rq1.empty:
                df_rq1.to_excel(writer, sheet_name="RQ1_Iterations", index=False)
            if not df_rq2.empty:
                df_rq2.to_excel(writer, sheet_name="RQ2_Complexity_Ceiling", index=False)
            if not df_rq3.empty:
                df_rq3.to_excel(writer, sheet_name="RQ3_Accuracy", index=False)
            per_task.to_excel(writer, sheet_name="Per_Task_Breakdown", index=False)
            if not df_ablate.empty:
                df_ablate.to_excel(writer, sheet_name="Router_Ablation", index=False)
        print(f"Generated multi-sheet Results.xlsx at: {os.path.relpath(excel_path, BASE_DIR)}")
    except Exception as e:
        print(f"Note: openpyxl writer notice: {e}")

    print(f"Generated granular per_task_results.csv at: {os.path.relpath(per_task_path, BASE_DIR)}")

if __name__ == "__main__":
    generate_master_results()
