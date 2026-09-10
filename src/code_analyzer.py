"""
Code Analyzer Agent (DeepSeek-Coder).
Generates unified diff patches from issue context and iterative QA feedback.
"""
import os
import time
import json
import requests
from typing import Dict, Any, List, Optional

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
import hashlib

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

def _task_hash(task_id: str) -> int:
    return int(hashlib.md5(task_id.encode("utf-8")).hexdigest()[:8], 16)

class CodeAnalyzerAgent:
    def __init__(
        self,
        model: str = "deepseek-coder:1.3b-instruct-q4_K_M",
        temperature: float = 0.2,
        top_p: float = 0.95
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def generate_patch(
        self,
        task_dir: str,
        router_result: Dict[str, Any],
        iteration: int = 1,
        qa_feedback: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ) -> tuple[str, int, int, float]:
        """
        Generates a candidate unified diff patch.
        Incorporates QA Verifier feedback if iteration > 1.
        """
        start_time = time.time()

        with open(os.path.join(task_dir, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        with open(os.path.join(task_dir, "issue.md"), "r", encoding="utf-8") as f:
            issue_text = f.read()

        target_files = router_result.get("target_files", meta.get("target_files", []))
        files_context = ""
        for rel_path in target_files:
            abs_path = os.path.join(task_dir, rel_path)
            if os.path.exists(abs_path):
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                files_context += f"\n--- File: {rel_path} ---\n{content}\n"

        prompt = f"""You are an expert Python software engineer repairing a defect.

{issue_text}

Router Analysis:
- Target Files: {target_files}
- Bug Hypothesis: {router_result.get('bug_hypothesis', 'N/A')}
- Recommended Focus: {router_result.get('recommended_focus', 'N/A')}

Source Code Context:
{files_context}
"""
        if iteration > 1 and qa_feedback:
            prompt += f"""
=== Previous Iteration Feedback (Iteration {iteration - 1}) ===
The previous patch failed pytest execution:
- Failed Tests: {qa_feedback.get('failed_tests', [])}
- Failure Diagnosis: {qa_feedback.get('root_cause_analysis', 'Assertion failure')}
- Actionable Repair Guidance: {qa_feedback.get('refinement_guidance', 'Fix logic to pass all assertions')}
- Traceback Snippet:
{qa_feedback.get('traceback_snippet', '')}
"""

        prompt += """
Please generate a single unified diff patch (starting with `--- a/...` and `+++ b/...`) that fixes the bug and passes all unit tests.
Output ONLY the unified diff block inside ```diff ... ``` code fences.
"""
        patch_text, p_tokens, c_tokens, latency = self._call_llm(prompt, meta, iteration, qa_feedback, seed)
        return patch_text, p_tokens, c_tokens, latency

    def _call_llm(
        self,
        prompt: str,
        meta: Dict[str, Any],
        iteration: int,
        qa_feedback: Optional[Dict[str, Any]],
        seed: int
    ) -> tuple[str, int, int, float]:
        """Calls Ollama API or falls back to calibrated iterative simulator."""
        start_time = time.time()
        if _is_ollama_online():
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": 1024
                    },
                    "keep_alive": 0 # Sequential fallback: unload model after invocation
                }
                resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    elapsed = time.time() - start_time
                    p_tokens = data.get("prompt_eval_count", len(prompt.split()) * 2)
                    c_tokens = data.get("eval_count", len(data.get("response", "").split()) * 2)
                    return data.get("response", ""), p_tokens, c_tokens, round(elapsed, 3)
            except Exception as e:
                raise RuntimeError(
                    f"Ollama request error for model '{self.model}': {e}. "
                    "Offline fallback has been permanently neutralized to prevent synthetic benchmark generation."
                )

        # Offline fallback is permanently neutralized
        raise RuntimeError(
            "Offline simulation fallback has been permanently neutralized in src/code_analyzer.py. "
            "Local Ollama server is offline or unreachable at http://localhost:11434. "
            "Please ensure Ollama is running, or use the active LangGraph swarm runner (src/swarm/swarm_runner.py)."
        )
