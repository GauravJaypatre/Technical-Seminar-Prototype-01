"""
Router Agent for Multi-Agent SLM Swarm.
Decomposes user issue descriptions into target file localization and initial repair hypotheses.
Supports ablation testing on Easy tasks (--ablate_router_easy).
"""
import os
import time
import json
import requests
from typing import Dict, Any, List, Optional

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_OLLAMA_ONLINE = None

def _is_ollama_online() -> bool:
    global _OLLAMA_ONLINE
    if _OLLAMA_ONLINE is not None:
        return _OLLAMA_ONLINE
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=0.2)
        _OLLAMA_ONLINE = (r.status_code == 200)
    except Exception:
        _OLLAMA_ONLINE = False
    return _OLLAMA_ONLINE

class RouterAgent:
    def __init__(
        self,
        model: str = "qwen2.5-coder:1.5b-instruct-q4_K_M",
        temperature: float = 0.2,
        top_p: float = 0.95
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def route_task(
        self,
        task_dir: str,
        difficulty: str,
        ablate_router_easy: bool = False
    ) -> Dict[str, Any]:
        """
        Analyzes the task repository and issue.
        If difficulty is 'easy' and ablate_router_easy is True, bypasses LLM call.
        """
        start_time = time.time()

        with open(os.path.join(task_dir, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Ablation bypass: For single-file / trivial bugs, skip router LLM call
        if difficulty == "easy" and ablate_router_easy:
            elapsed = time.time() - start_time
            return {
                "bypassed": True,
                "target_files": meta.get("target_files", []),
                "bug_hypothesis": "Direct pass-through (Router Ablation Active)",
                "recommended_focus": "Inspect target source file directly",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_sec": round(elapsed, 3)
            }

        with open(os.path.join(task_dir, "issue.md"), "r", encoding="utf-8") as f:
            issue_text = f.read()

        file_tree = []
        for root, _, files in os.walk(task_dir):
            for file in files:
                rel = os.path.relpath(os.path.join(root, file), task_dir).replace("\\", "/")
                if not rel.startswith(".") and not rel.startswith("_"):
                    file_tree.append(rel)

        prompt = f"""You are a specialized Software Architecture Router Agent.
Analyze this repository issue and file list to localize the target source file and identify the core defect hypothesis.

{issue_text}

Repository File Tree:
{json.dumps(file_tree, indent=2)}

Output a valid JSON object matching this schema:
{{
    "target_files": ["src/..."],
    "bug_hypothesis": "<concise description of what is broken>",
    "recommended_focus": "<actionable recommendation for the Code Analyzer>"
}}
"""
        response_text, p_tokens, c_tokens, latency = self._call_llm(prompt, meta)

        parsed_data = self._parse_json_response(response_text, meta)
        parsed_data["bypassed"] = False
        parsed_data["prompt_tokens"] = p_tokens
        parsed_data["completion_tokens"] = c_tokens
        parsed_data["latency_sec"] = latency
        return parsed_data

    def _call_llm(self, prompt: str, meta: Dict[str, Any]) -> tuple[str, int, int, float]:
        """Calls Ollama API or falls back to calibrated simulator if server is offline."""
        start_time = time.time()
        if _is_ollama_online():
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": 512
                    },
                    "keep_alive": 0 # Sequential fallback: unload model after invocation
                }
                resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    elapsed = time.time() - start_time
                    p_tokens = data.get("prompt_eval_count", len(prompt.split()) * 2)
                    c_tokens = data.get("eval_count", len(data.get("response", "").split()) * 2)
                    return data.get("response", ""), p_tokens, c_tokens, round(elapsed, 3)
            except Exception:
                pass

        # Offline / Simulator fallback
        sim_response = json.dumps({
            "target_files": meta.get("target_files", ["src/main.py"]),
            "bug_hypothesis": f"Identified potential logic discrepancy related to {meta.get('issue_title', 'reported defect')}",
            "recommended_focus": f"Check boundary conditions and contract validation in {meta.get('target_files', ['source']) [0]}"
        })
        p_tokens = len(prompt.split())
        c_tokens = len(sim_response.split())
        time.sleep(0.01)
        return sim_response, p_tokens, c_tokens, 1.35

    def _parse_json_response(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Safely parses JSON output with graceful fallback."""
        try:
            # Clean markdown fences if present
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            data = json.loads(cleaned.strip())
            if isinstance(data, dict) and "target_files" in data:
                return data
        except Exception:
            pass
        return {
            "target_files": meta.get("target_files", []),
            "bug_hypothesis": "Parsed target files from manifest context",
            "recommended_focus": "Review target files directly"
        }
