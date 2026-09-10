"""
Multi-SLM Swarm Execution Harness:
Integrates the LangGraph Swarm workflow (Router + Code Analyzer + QA Verifier)
with the standard Micro-SWE benchmark harness and sandbox evaluation pipeline.
Outputs results cleanly into results/prototype_run_swarm/.
"""
import os
import sys
import time
import json
import argparse
import datetime
from typing import Optional, List, Dict, Any, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.sandbox import run_in_sandbox
from src.benchmarking_extended import BenchmarkLogger, ResourceMonitor
from src.swarm.graph import build_swarm_graph, SwarmState
from src.swarm.agents import OllamaClient


class SwarmBenchmarkLogger:
    """
    Incremental checkpointing logger for Multi-SLM Swarm benchmark.
    Immediately flushes each completed run's record to JSON and CSV on completion.
    """
    def __init__(self, output_dir: str, session_name: str = "swarm"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = os.path.join(self.output_dir, f"benchmark_prototype_{session_name}_{timestamp}.json")
        self.csv_path = os.path.join(self.output_dir, f"benchmark_prototype_{session_name}_{timestamp}.csv")
        self.runs: List[Dict[str, Any]] = []
        self.csv_headers = [
            "run_id", "timestamp", "task_id", "difficulty", "system", "model_profile",
            "provider", "model_name", "is_prototype", "is_live",
            "seed", "temperature", "top_p", "pass_at_1", "final_pass", "iterations_to_success",
            "total_iterations_run", "cumulative_wall_clock_sec", "total_prompt_tokens",
            "total_completion_tokens", "estimated_cost_usd", "peak_ram_mb", "used_fallback",
            "qa_exhausted", "qa_retries", "router_sub_steps", "agents_involved"
        ]
        with open(self.csv_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(",".join(self.csv_headers) + "\n")

    def log_run(self, run_record: Dict[str, Any]):
        self.runs.append(run_record)
        # Immediate atomic flush to JSON
        with open(self.json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.runs, f, indent=2)
            f.flush()

        # Immediate append to CSV with flush
        row_vals = [
            str(run_record.get("run_id", "")),
            str(run_record.get("timestamp", "")),
            str(run_record.get("task_id", "")),
            str(run_record.get("difficulty", "")),
            str(run_record.get("system", "")),
            str(run_record.get("model_profile", "")),
            str(run_record.get("provider", "")),
            str(run_record.get("model_name", "")),
            str(run_record.get("is_prototype", True)),
            str(run_record.get("is_live", True)),
            str(run_record.get("seed", "")),
            str(run_record.get("temperature", "")),
            str(run_record.get("top_p", "")),
            str(int(run_record.get("pass_at_1", False))),
            str(int(run_record.get("final_pass", False))),
            str(run_record.get("iterations_to_success", "")),
            str(run_record.get("total_iterations_run", "")),
            str(run_record.get("cumulative_wall_clock_sec", 0.0)),
            str(run_record.get("total_prompt_tokens", 0)),
            str(run_record.get("total_completion_tokens", 0)),
            str(run_record.get("estimated_cost_usd", 0.0)),
            str(run_record.get("peak_ram_mb", 0.0)),
            str(bool(run_record.get("used_fallback", False))),
            str(bool(run_record.get("qa_exhausted", False))),
            str(run_record.get("qa_retries", 0)),
            str(run_record.get("router_sub_steps", 1)),
            '"' + ";".join(run_record.get("agents_involved", [])) + '"'
        ]
        with open(self.csv_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(",".join(row_vals) + "\n")
            f.flush()


def run_swarm_baseline(
    live: bool = True,
    analyzer_model: str = "deepseek-coder:6.7b",
    verifier_model: str = "qwen2.5:0.5b",
    seeds: Optional[List[int]] = None,
    task_limit: Optional[int] = None,
    task_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Tuple[str, str]:
    """
    Executes the Multi-SLM Swarm baseline benchmark across Micro-SWE tasks and seeds.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "results", "prototype_run_swarm")
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(BASE_DIR, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Enforce live execution requirement
    if not live:
        raise RuntimeError(
            "Offline simulation mode has been permanently neutralized in src/swarm/swarm_runner.py. "
            "A live API connection is strictly required. Pass `live=True` or run with `--live`."
        )

    # Prevent Windows from sleeping/suspending during long unattended benchmark
    if sys.platform == "win32":
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_AWAYMODE_REQUIRED = 0x00000002
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        except Exception:
            pass

    # Compile LangGraph workflow
    ollama_client = OllamaClient()
    app = build_swarm_graph(
        client=ollama_client,
        analyzer_model=analyzer_model,
        verifier_model=verifier_model
    )

    # Load tasks from canonical task manifest (exact same source of truth)
    manifest_path = os.path.join(BASE_DIR, "data", "task_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks = manifest["all_tasks"]
    if task_ids:
        tasks = [t for t in tasks if t["task_id"] in task_ids]
    if task_limit:
        tasks = tasks[:task_limit]

    total_runs = len(tasks) * len(seeds)
    print(f"\n========================================================")
    print(f"Starting Multi-SLM Swarm Benchmark [PROTOTYPE RUN]")
    print(f"Workflow: LangGraph (Router -> DeepSeek-Coder -> Qwen 2.5 QA)")
    print(f"Live Local Inference: {live} (Analyzer: {analyzer_model}, Verifier: {verifier_model})")
    print(f"Output Path: {os.path.relpath(output_dir, BASE_DIR)}")
    print(f"Total Tasks: {len(tasks)} | Seeds: {seeds} | Total Runs: {total_runs}")
    print(f"========================================================\n")

    logger = SwarmBenchmarkLogger(
        output_dir=output_dir,
        session_name="swarm"
    )
    monitor = ResourceMonitor(interval=0.2)
    run_counter = 0

    for task_info in tasks:
        task_id = task_info["task_id"]
        tier = task_info["tier"]
        task_dir = os.path.join(BASE_DIR, task_info["directory"])

        with open(os.path.join(task_dir, "issue.md"), "r", encoding="utf-8") as f:
            issue_text = f.read()

        with open(os.path.join(task_dir, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        target_files = meta.get("target_files", [])

        for seed in seeds:
            run_counter += 1
            run_id = f"proto_swarm_{task_id}_s{seed}"
            print(f"\n[{run_counter}/{total_runs}] Task: {task_id} ({tier}) | Seed: {seed}")

            monitor.start()
            run_start_time = time.time()

            initial_state: SwarmState = {
                "task_id": task_id,
                "tier": tier,
                "task_dir": task_dir,
                "issue_text": issue_text,
                "target_files": target_files,
                "seed": seed,
                "temperature": 0.2,
                "routing_decision": {},
                "candidate_patch": "",
                "previous_patch": None,
                "qa_critique": None,
                "retry_count": 0,
                "is_valid": False,
                "qa_exhausted": False,
                "final_patch": "",
                "agents_involved": [],
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_latency_sec": 0.0
            }

            try:
                final_state = app.invoke(initial_state)
                patch = final_state.get("final_patch") or final_state.get("candidate_patch", "")
                p_tokens = final_state.get("total_prompt_tokens", 0)
                c_tokens = final_state.get("total_completion_tokens", 0)
                agents = final_state.get("agents_involved", ["router", "code_analyzer", "qa_verifier"])
                router_sub_steps = final_state.get("routing_decision", {}).get("sub_steps", 1)
                qa_retries = final_state.get("retry_count", 0)
                qa_exhausted = final_state.get("qa_exhausted", False)
            except Exception as e:
                print(f"[ERROR] Task {task_id} seed {seed} Swarm invocation failed: {e}")
                patch = f"# SWARM INVOCATION ERROR: {e}"
                p_tokens = 0
                c_tokens = 0
                agents = ["error"]
                router_sub_steps = 1
                qa_retries = 0
                qa_exhausted = False

            # Sandbox evaluation
            sandbox_res = run_in_sandbox(task_dir, patch_str=patch)
            total_wall_clock = time.time() - run_start_time
            peak_ram, peak_cpu = monitor.stop()

            run_record = {
                "run_id": run_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "task_id": task_id,
                "difficulty": tier,
                "system": "multi_slm_swarm",
                "model_profile": f"{analyzer_model}+{verifier_model}",
                "provider": "ollama_local",
                "model_name": analyzer_model,
                "is_prototype": True,
                "is_live": live,
                "seed": seed,
                "temperature": 0.2,
                "top_p": 0.95,
                "pass_at_1": sandbox_res.success,
                "final_pass": sandbox_res.success,
                "iterations_to_success": 1 if sandbox_res.success else None,
                "total_iterations_run": 1 + qa_retries,
                "cumulative_wall_clock_sec": round(total_wall_clock, 2),
                "total_prompt_tokens": p_tokens,
                "total_completion_tokens": c_tokens,
                "estimated_cost_usd": 0.0,
                "peak_ram_mb": peak_ram,
                "used_fallback": sandbox_res.used_fallback,
                "qa_exhausted": qa_exhausted,
                "qa_retries": qa_retries,
                "router_sub_steps": router_sub_steps,
                "agents_involved": agents,
                "raw_patch": patch,
                "sandbox_detail": sandbox_res.to_dict()
            }

            logger.log_run(run_record)

    rel_json = os.path.relpath(logger.json_path, BASE_DIR).replace("\\", "/")
    rel_csv = os.path.relpath(logger.csv_path, BASE_DIR).replace("\\", "/")
    print(f"\nMulti-SLM Swarm benchmark run completed!")
    print(f"Results saved to:\n  JSON: {rel_json}\n  CSV:  {rel_csv}")
    return logger.json_path, logger.csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-SLM Swarm Baseline Benchmark")
    parser.add_argument("--live", action="store_true", default=True, help="Execute live local Ollama SLMs")
    parser.add_argument("--mock", action="store_true", help="Execute mock run for offline validation")
    parser.add_argument("--analyzer", type=str, default="deepseek-coder:6.7b", help="Analyzer model name")
    parser.add_argument("--verifier", type=str, default="qwen2.5:0.5b", help="Verifier model name")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Filter specific task IDs")
    parser.add_argument("--limit", type=int, default=None, help="Limit task count")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()
    if args.mock:
        raise RuntimeError(
            "Offline mock mode has been permanently neutralized in src/swarm/swarm_runner.py. "
            "Execute with --live against running Ollama service."
        )

    run_swarm_baseline(
        live=args.live,
        analyzer_model=args.analyzer,
        verifier_model=args.verifier,
        seeds=args.seeds,
        task_limit=args.limit,
        task_ids=args.tasks,
        output_dir=args.output_dir
    )
