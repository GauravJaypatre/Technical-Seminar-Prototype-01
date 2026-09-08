"""
Incremental benchmarking logger and system resource monitor.
Extends the reference implementation from reference_slm_study/benchmarking.py.
Features:
- Incremental JSON and CSV persistence after every single task run.
- Background peak RAM and CPU monitoring via psutil.
- Standardized schema for both Monolithic and Swarm runs.
"""
import os
import sys
import time
import json
import psutil
import datetime
import threading
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_RUNS_DIR = os.path.join(BASE_DIR, "results", "runs")
os.makedirs(RESULTS_RUNS_DIR, exist_ok=True)

class ResourceMonitor:
    """Monitors peak host RAM (in MB) and CPU percent in background thread."""
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_ram_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.thread: Optional[threading.Thread] = None

    def _monitor(self):
        process = psutil.Process(os.getpid())
        while not self.stop_event.is_set():
            try:
                # Include child processes (sandbox executions) in memory footprint
                mem = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        mem += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                ram_mb = mem / (1024 * 1024)
                if ram_mb > self.peak_ram_mb:
                    self.peak_ram_mb = ram_mb

                cpu = psutil.cpu_percent(interval=None)
                if cpu > self.peak_cpu_percent:
                    self.peak_cpu_percent = cpu
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_ram_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> tuple[float, float]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        return round(self.peak_ram_mb, 2), round(self.peak_cpu_percent, 2)

class BenchmarkLogger:
    """Manages incremental run persistence to JSON and CSV."""
    def __init__(
        self,
        session_name: str = "run",
        output_dir: Optional[str] = None,
        is_prototype: bool = False
    ):
        self.output_dir = output_dir or RESULTS_RUNS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.is_prototype = is_prototype

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "benchmark_prototype" if is_prototype else "benchmark"
        self.json_path = os.path.join(self.output_dir, f"{prefix}_{session_name}_{timestamp}.json")
        self.csv_path = os.path.join(self.output_dir, f"{prefix}_{session_name}_{timestamp}.csv")
        self.runs: List[Dict[str, Any]] = []
        self._init_csv()

    def _init_csv(self):
        if self.is_prototype:
            headers = [
                "run_id", "timestamp", "task_id", "difficulty", "system", "model_profile",
                "provider", "model_name", "is_prototype",
                "seed", "temperature", "top_p", "pass_at_1", "final_pass", "iterations_to_success",
                "total_iterations_run", "cumulative_wall_clock_sec", "total_prompt_tokens",
                "total_completion_tokens", "estimated_cost_usd", "peak_ram_mb", "used_fallback"
            ]
        else:
            headers = [
                "run_id", "timestamp", "task_id", "difficulty", "system", "model_profile",
                "seed", "temperature", "top_p", "pass_at_1", "final_pass", "iterations_to_success",
                "total_iterations_run", "cumulative_wall_clock_sec", "total_prompt_tokens",
                "total_completion_tokens", "estimated_cost_usd", "peak_ram_mb", "used_fallback"
            ]
        with open(self.csv_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(",".join(headers) + "\n")

    def log_run(self, run_record: Dict[str, Any]):
        """Appends run record immediately to JSON and CSV files."""
        self.runs.append(run_record)
        # 1. Update JSON
        with open(self.json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.runs, f, indent=2)

        # 2. Append to CSV
        if self.is_prototype:
            csv_row = [
                str(run_record.get("run_id", "")),
                str(run_record.get("timestamp", "")),
                str(run_record.get("task_id", "")),
                str(run_record.get("difficulty", "")),
                str(run_record.get("system", "")),
                str(run_record.get("model_profile", "")),
                str(run_record.get("provider", "")),
                str(run_record.get("model_name", "")),
                str(run_record.get("is_prototype", True)),
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
                str(bool(run_record.get("used_fallback", False)))
            ]
        else:
            csv_row = [
                str(run_record.get("run_id", "")),
                str(run_record.get("timestamp", "")),
                str(run_record.get("task_id", "")),
                str(run_record.get("difficulty", "")),
                str(run_record.get("system", "")),
                str(run_record.get("model_profile", "")),
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
                str(bool(run_record.get("used_fallback", False)))
            ]
        with open(self.csv_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(",".join(csv_row) + "\n")

        proto_tag = " [PROTOTYPE]" if self.is_prototype else ""
        fb_tag = " (fallback)" if run_record.get("used_fallback") else ""
        print(f"Logged run {run_record.get('run_id')}{proto_tag}{fb_tag}: Success={run_record.get('final_pass')} (iter={run_record.get('iterations_to_success')}) in {run_record.get('cumulative_wall_clock_sec')}s")

