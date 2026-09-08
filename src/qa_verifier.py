"""
QA Verifier Agent (Qwen2.5-Coder + Sandbox Execution).
Executes candidate patches in the isolated sandbox, and upon failure, synthesizes
structured root-cause diagnostics and repair hints for the Code Analyzer.
"""
import os
import time
import json
import requests
from typing import Dict, Any, List, Optional
from src.sandbox import run_in_sandbox, SandboxResult

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

class QAVerifierAgent:
    def __init__(
        self,
        model: str = "qwen2.5-coder:1.5b-instruct-q4_K_M",
        temperature: float = 0.2,
        top_p: float = 0.95
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def verify_patch(
        self,
        task_dir: str,
        patch_str: str,
        timeout_sec: float = 15.0
    ) -> tuple[SandboxResult, Optional[Dict[str, Any]], int, int, float]:
        """
        Executes patch in sandbox. If tests fail, invokes Qwen to synthesize structured feedback.
        Returns: (sandbox_result, structured_feedback, p_tokens, c_tokens, verifier_latency)
        """
        start_time = time.time()
        sandbox_res = run_in_sandbox(task_dir, patch_str=patch_str, timeout_sec=timeout_sec)

        # If tests passed completely, no feedback needed
        if sandbox_res.success:
            elapsed = time.time() - start_time
            return sandbox_res, None, 0, 0, round(elapsed, 3)

        # Pytest failed: QA Verifier analyzes failure trace
        llm_start = time.time()
        feedback, p_tokens, c_tokens = self._generate_feedback(sandbox_res)
        verifier_latency = time.time() - llm_start

        return sandbox_res, feedback, p_tokens, c_tokens, round(verifier_latency, 3)

    def _generate_feedback(self, sandbox_res: SandboxResult) -> tuple[Dict[str, Any], int, int]:
        """Invokes Qwen model to analyze failure logs and output structured guidance."""
        failure_log = (sandbox_res.stdout + "\n" + sandbox_res.stderr).strip()
        if len(failure_log) > 3000:
            failure_log = failure_log[-3000:]

        prompt = f"""You are a senior QA Test Engineer and Verifier.
The proposed patch failed during pytest execution.

Test Output:
{failure_log}

Failed Tests: {sandbox_res.failed_tests}
Patch Application Status: {sandbox_res.patch_applied} (Error: {sandbox_res.patch_error})

Provide a structured diagnostic JSON matching this schema:
{{
    "failed_tests": {json.dumps(sandbox_res.failed_tests)},
    "failure_type": "<e.g. AssertionError, KeyError, Timeout, PatchRejection>",
    "root_cause_analysis": "<concise technical explanation of why the patch failed the test assertions>",
    "refinement_guidance": "<specific code change advice for the Code Analyzer to fix the failure>",
    "traceback_snippet": "<key 3-5 line failure traceback excerpt>"
}}
"""
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
                    p_tokens = data.get("prompt_eval_count", len(prompt.split()) * 2)
                    c_tokens = data.get("eval_count", len(data.get("response", "").split()) * 2)
                    parsed = json.loads(data.get("response", "{}"))
                    return parsed, p_tokens, c_tokens
            except Exception:
                pass

        # Offline fallback diagnostic generator
        p_tokens = len(prompt.split())
        err_snippet = failure_log[-400:] if failure_log else "Test assertion failed"
        feedback = {
            "failed_tests": sandbox_res.failed_tests or ["test_failure"],
            "failure_type": "AssertionError" if "AssertionError" in failure_log else "TestExecutionFailure",
            "root_cause_analysis": "The patched code failed assertion checks in pytest.",
            "refinement_guidance": "Review the failing test assertions and ensure all edge cases and return types match specifications.",
            "traceback_snippet": err_snippet
        }
        c_tokens = len(json.dumps(feedback).split())
        return feedback, p_tokens, c_tokens
