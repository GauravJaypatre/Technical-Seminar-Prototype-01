"""
Multi-Agent SLM Swarm Pipeline.
Orchestrates:
1. Router Agent (qwen2.5-coder:1.5b/7b)
2. Code Analyzer (deepseek-coder:1.3b/6.7b)
3. QA Verifier (qwen2.5-coder:1.5b/7b + sandbox.py)

Controls iterative self-correction loop up to K=5 iterations per task across 3 seeds (42, 43, 44).
Supports router ablation on Easy tasks (--ablate_router_easy).
Logs granular incremental metrics via BenchmarkLogger.
"""
import os
import sys
import time
import json
import argparse
import datetime
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.router import RouterAgent
from src.code_analyzer import CodeAnalyzerAgent
from src.qa_verifier import QAVerifierAgent
from src.benchmarking_extended import BenchmarkLogger, ResourceMonitor

# Estimated compute cost per 1M tokens for local SLMs (energy / compute equivalent)
SLM_COMPUTE_COST_PER_MILLION = 0.50

def run_swarm_pipeline(
    profile: str = "lightweight",
    max_k: int = 5,
    seeds: Optional[List[int]] = None,
    task_limit: Optional[int] = None,
    ablate_router_easy: bool = False,
    session_suffix: str = ""
):
    """Executes the multi-agent SLM swarm benchmark."""
    if seeds is None:
        seeds = [42, 43, 44] # 3 seeds for swarm as approved

    # Pinned model tags
    if profile == "standard":
        router_model = "qwen2.5-coder:7b-instruct-q4_K_M"
        analyzer_model = "deepseek-coder:6.7b-instruct-q4_K_M"
        verifier_model = "qwen2.5-coder:7b-instruct-q4_K_M"
    else: # lightweight / primary pilot
        router_model = "qwen2.5-coder:1.5b-instruct-q4_K_M"
        analyzer_model = "deepseek-coder:1.3b-instruct-q4_K_M"
        verifier_model = "qwen2.5-coder:1.5b-instruct-q4_K_M"

    router = RouterAgent(model=router_model, temperature=0.2, top_p=0.95)
    analyzer = CodeAnalyzerAgent(model=analyzer_model, temperature=0.2, top_p=0.95)
    verifier = QAVerifierAgent(model=verifier_model, temperature=0.2, top_p=0.95)

    manifest_path = os.path.join(BASE_DIR, "data", "task_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks = manifest["all_tasks"]
    if task_limit:
        tasks = tasks[:task_limit]

    session_name = f"swarm_{profile}"
    if ablate_router_easy:
        session_name += "_ablate_easy"
    if session_suffix:
        session_name += f"_{session_suffix}"

    logger = BenchmarkLogger(session_name=session_name)
    monitor = ResourceMonitor(interval=0.2)

    print(f"\n========================================================")
    print(f"Starting Multi-Agent SLM Swarm Benchmark")
    print(f"Profile: {profile} | Max Iterations K: {max_k}")
    print(f"Router Model:   {router_model}")
    print(f"Analyzer Model: {analyzer_model}")
    print(f"Verifier Model: {verifier_model}")
    print(f"Seeds: {seeds} | Tasks: {len(tasks)} | Total Runs: {len(tasks) * len(seeds)}")
    print(f"Router Ablation on Easy: {ablate_router_easy}")
    print(f"========================================================\n")

    for task_info in tasks:
        task_id = task_info["task_id"]
        tier = task_info["tier"]
        task_dir = os.path.join(BASE_DIR, task_info["directory"])

        for seed in seeds:
            run_id = f"swarm_{profile}_{task_id}_s{seed}"
            monitor.start()
            run_start_time = time.time()

            total_prompt_tokens = 0
            total_completion_tokens = 0
            iteration_history = []
            final_pass = False
            pass_at_1 = False
            iterations_to_success = None
            qa_feedback = None

            # 1. Routing phase
            route_info = router.route_task(
                task_dir=task_dir,
                difficulty=tier,
                ablate_router_easy=ablate_router_easy
            )
            total_prompt_tokens += route_info.get("prompt_tokens", 0)
            total_completion_tokens += route_info.get("completion_tokens", 0)

            # 2. Iterative refinement loop: k = 1..max_k
            total_iterations_run = 0
            for k in range(1, max_k + 1):
                total_iterations_run = k
                iter_start = time.time()

                # Generate patch via Code Analyzer
                patch, a_pt, a_ct, a_lat = analyzer.generate_patch(
                    task_dir=task_dir,
                    router_result=route_info,
                    iteration=k,
                    qa_feedback=qa_feedback,
                    seed=seed
                )
                total_prompt_tokens += a_pt
                total_completion_tokens += a_ct

                # Verify patch via QA Verifier & Sandbox
                sandbox_res, feedback, v_pt, v_ct, v_lat = verifier.verify_patch(
                    task_dir=task_dir,
                    patch_str=patch,
                    timeout_sec=15.0
                )
                total_prompt_tokens += v_pt
                total_completion_tokens += v_ct

                iter_record = {
                    "iteration": k,
                    "analyzer_latency_sec": a_lat,
                    "verifier_latency_sec": v_lat,
                    "sandbox_exit_code": sandbox_res.exit_code,
                    "tests_passed": sandbox_res.success,
                    "passed_tests": sandbox_res.passed_tests,
                    "failed_tests": sandbox_res.failed_tests,
                    "patch_applied": sandbox_res.patch_applied,
                    "iteration_elapsed_sec": round(time.time() - iter_start, 2)
                }
                iteration_history.append(iter_record)

                if k == 1:
                    pass_at_1 = sandbox_res.success

                if sandbox_res.success:
                    final_pass = True
                    iterations_to_success = k
                    break
                else:
                    qa_feedback = feedback

            total_wall_clock = time.time() - run_start_time
            peak_ram, peak_cpu = monitor.stop()

            # Cost estimation
            total_tokens = total_prompt_tokens + total_completion_tokens
            cost_usd = total_tokens * 1e-6 * SLM_COMPUTE_COST_PER_MILLION

            system_label = "swarm_slm_no_router" if (tier == "easy" and ablate_router_easy) else f"swarm_slm_{profile}"
            run_record = {
                "run_id": run_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "task_id": task_id,
                "difficulty": tier,
                "system": system_label,
                "model_profile": f"{analyzer_model}+{verifier_model}",
                "seed": seed,
                "temperature": 0.2,
                "top_p": 0.95,
                "router_enabled": not (tier == "easy" and ablate_router_easy),
                "pass_at_1": pass_at_1,
                "final_pass": final_pass,
                "iterations_to_success": iterations_to_success,
                "resolved_at_iteration": iterations_to_success, # None if failed within K=5
                "total_iterations_run": total_iterations_run,
                "cumulative_wall_clock_sec": round(total_wall_clock, 2),
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "estimated_cost_usd": round(cost_usd, 6),
                "peak_ram_mb": peak_ram,
                "peak_cpu_percent": peak_cpu,
                "iteration_history": iteration_history
            }

            logger.log_run(run_record)

    rel_json = os.path.relpath(logger.json_path, BASE_DIR).replace("\\", "/")
    rel_csv = os.path.relpath(logger.csv_path, BASE_DIR).replace("\\", "/")
    print(f"\nSwarm benchmark completed successfully!")
    print(f"Results saved to:\n  JSON: {rel_json}\n  CSV:  {rel_csv}")
    return logger.json_path, logger.csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Agent SLM Swarm Benchmark")
    parser.add_argument("--profile", choices=["lightweight", "standard"], default="lightweight", help="Model profile")
    parser.add_argument("--max_k", type=int, default=5, help="Maximum refinement iterations")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds (3 recommended)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--ablate_router_easy", action="store_true", help="Bypass router for easy tasks")
    parser.add_argument("--suffix", type=str, default="", help="Optional session filename suffix")
    args = parser.parse_args()

    run_swarm_pipeline(
        profile=args.profile,
        max_k=args.max_k,
        seeds=args.seeds,
        task_limit=args.limit,
        ablate_router_easy=args.ablate_router_easy,
        session_suffix=args.suffix
    )
