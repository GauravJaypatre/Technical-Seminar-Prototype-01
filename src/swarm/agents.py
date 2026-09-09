"""
Swarm Agents Module:
Implements the three specialized agents of the Multi-SLM Swarm:
1. Router Agent (Deterministic Heuristics - zero token cost, reproducible across seeds).
2. Code Analyzer SLM (DeepSeek-Coder:6.7b - unified diff patch generation).
3. QA Verifier SLM (Qwen 2.5:0.5b - syntax, header, and hallucination inspection).

Communicates with local Ollama service via OpenAI-compatible REST endpoints.
"""
import os
import sys
import time
import json
import logging
from typing import Optional, Dict, Any, List, Tuple

import requests
from src.baseline_client import GenerationResult
from src.sandbox import clean_diff_text

logger = logging.getLogger("swarm_agents")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class OllamaClient:
    """
    Lightweight OpenAI-compatible client for local Ollama server.
    Default endpoint: http://127.0.0.1:11434/v1
    """
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        timeout_sec: float = 180.0
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        for k, v in kwargs.items():
            if k not in payload:
                payload[k] = v

        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            elapsed = time.time() - start_time

            if response.status_code != 200:
                err_msg = f"Ollama HTTP {response.status_code} ({url}): {response.text[:300]}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError(f"Empty choices from Ollama model '{model}': {data}")

            msg = choices[0].get("message", {})
            content = msg.get("content") or ""

            usage = data.get("usage", {})
            p_tokens = usage.get("prompt_tokens", 0)
            c_tokens = usage.get("completion_tokens", 0)

            return GenerationResult(
                text=content,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                latency_sec=elapsed,
                provider="ollama_local",
                model_name=model,
                raw_response=data
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Ollama request error ({model}): {e}")
            raise


class RouterAgent:
    """
    Router Agent: Deterministic Heuristic-Based Orchestration.
    
    Locked Heuristic Rules:
    - Zero LLM calls: keeps routing 100% deterministic and reproducible across seeds.
    - Zero token cost: ensures Router overhead does not inflate SLM latency or token totals.
    
    Exact Decision Thresholds:
    1. Tier: 'easy':
       - If single target file AND len(issue_text) < 1500 chars -> 'direct' strategy (sub_steps = 1).
       - Else -> 'decomposed' strategy (sub_steps = 2).
    2. Tier: 'medium':
       - If multi-file (len(target_files) > 1) OR len(issue_text) >= 1500 chars -> 'decomposed' (sub_steps = 2).
       - Else -> 'direct' (sub_steps = 1).
    3. Tier: 'hard':
       - Always 'decomposed' (sub_steps = 2): multi-step context partitioning and localization guidance.
    """
    def route(
        self,
        tier: str,
        target_files: List[str],
        issue_text: str
    ) -> Dict[str, Any]:
        tier_lower = (tier or "medium").lower().strip()
        num_files = len(target_files)
        issue_len = len(issue_text)

        if tier_lower == "easy":
            if num_files <= 1 and issue_len < 1500:
                strategy = "direct"
                sub_steps = 1
                reason = f"Easy tier with single file ({num_files}) and concise issue ({issue_len} chars): direct patch generation."
            else:
                strategy = "decomposed"
                sub_steps = 2
                reason = f"Easy tier with multiple files ({num_files}) or extended issue ({issue_len} chars): decomposed localization."

        elif tier_lower == "medium":
            if num_files > 1 or issue_len >= 1500:
                strategy = "decomposed"
                sub_steps = 2
                reason = f"Medium tier with {num_files} files and {issue_len} chars: decomposed localization before diff generation."
            else:
                strategy = "direct"
                sub_steps = 1
                reason = f"Medium tier with single file and concise issue: direct generation."

        else:  # hard
            strategy = "decomposed"
            sub_steps = 2
            reason = f"Hard tier task: always requires two-step decomposition (localization + patch synthesis)."

        return {
            "strategy": strategy,
            "sub_steps": sub_steps,
            "reason": reason,
            "focus_files": target_files
        }


class CodeAnalyzerAgent:
    """
    Code Analyzer SLM Agent: DeepSeek-Coder:6.7b.
    Generates unified diff patch for the identified bug.
    """
    def __init__(
        self,
        client: OllamaClient,
        model_name: str = "deepseek-coder:6.7b"
    ):
        self.client = client
        self.model_name = model_name

    def generate_patch(
        self,
        task_dir: str,
        issue_text: str,
        target_files: List[str],
        routing_decision: Dict[str, Any],
        qa_critique: Optional[str] = None,
        previous_patch: Optional[str] = None,
        seed: int = 42,
        temperature: float = 0.2
    ) -> Tuple[str, int, int, float]:
        """
        Generates unified diff patch.
        Returns (cleaned_patch, prompt_tokens, completion_tokens, latency_sec).
        """
        prompt = f"""You are an expert software engineer resolving a bug in a repository.

{issue_text}

Here are the target files in the repository:
"""
        for rel_path in target_files:
            file_path = os.path.join(task_dir, rel_path)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                prompt += f"\n--- File: {rel_path} ---\n{content}\n"

        # If decomposed strategy, add explicit localization directive
        if routing_decision.get("strategy") == "decomposed":
            prompt += f"""
NOTE: The Router has identified this as a decomposed task. Focus modifications specifically on: {', '.join(routing_decision.get('focus_files', target_files))}.
"""

        # If QA critique provided from prior failed attempt, append retry feedback
        if qa_critique:
            prompt += f"""
CRITICAL QA VERIFIER FEEDBACK FROM PREVIOUS ATTEMPT:
{qa_critique}
Previous attempted patch:
```diff
{previous_patch}
```
Please address all above issues and regenerate a valid, well-formed patch.
"""

        prompt += """
Please generate a single unified diff patch (starting with `--- a/...` and `+++ b/...`) that resolves the issue and passes all existing regression tests.
Output ONLY the unified diff block inside ```diff ... ``` code fences.
"""
        messages = [{"role": "user", "content": prompt}]
        res = self.client.generate(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
            seed=seed
        )

        cleaned_patch = clean_diff_text(res.text)
        return cleaned_patch, res.prompt_tokens, res.completion_tokens, res.latency_sec


class QAVerifierAgent:
    """
    QA Verifier SLM Agent: Qwen 2.5:0.5b.
    Reviews proposed candidate patch before sandbox execution:
    - Sanity-checks diff format (fences, headers, hunk markers).
    - Checks for hallucinated or mismatched file paths.
    - Evaluates patch validity and provides critique on syntax errors.
    """
    def __init__(
        self,
        client: OllamaClient,
        model_name: str = "qwen2.5:0.5b"
    ):
        self.client = client
        self.model_name = model_name

    def verify_patch(
        self,
        candidate_patch: str,
        target_files: List[str],
        issue_text: str,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Inspects candidate patch.
        Returns dict with keys:
          - is_valid: bool
          - critique: Optional[str]
          - tokens: int
          - latency: float
        """
        # 1. Deterministic Structural Pre-Validation
        patch_str = candidate_patch.strip()
        if not patch_str or patch_str.startswith("# API ERROR"):
            return {
                "is_valid": False,
                "critique": "Candidate patch is empty or represents an API failure.",
                "tokens": 0,
                "latency": 0.0
            }

        # Check for unified diff headers
        has_orig_header = "--- a/" in patch_str or "--- " in patch_str
        has_new_header = "+++ b/" in patch_str or "+++ " in patch_str
        if not (has_orig_header and has_new_header):
            return {
                "is_valid": False,
                "critique": "Missing unified diff headers (`--- a/...` and `+++ b/...`). Ensure standard diff format.",
                "tokens": 0,
                "latency": 0.0
            }

        # Check for hallucinated file paths
        referenced_files = []
        for line in patch_str.splitlines():
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                ref = line[6:].strip()
                referenced_files.append(ref)

        normalized_targets = [f.replace("\\", "/") for f in target_files]
        hallucinated = [rf for rf in referenced_files if rf not in normalized_targets and not any(rf.endswith(tf) for tf in normalized_targets)]
        if hallucinated:
            return {
                "is_valid": False,
                "critique": f"Patch targets non-existent/hallucinated files: {hallucinated}. Target files must be from: {normalized_targets}.",
                "tokens": 0,
                "latency": 0.0
            }

        # 2. SLM Consistency Verification (Qwen 2.5:0.5b)
        verification_prompt = f"""You are a strict QA verification engineer reviewing an automated code patch.
Target files allowed: {target_files}

Candidate Patch:
{candidate_patch[:1500]}

Verify that:
1. The patch contains valid unified diff hunk headers (@@ ... @@ or @@).
2. It modifies allowed target files and looks like a coherent code modification.

Reply ONLY with a JSON object in this exact format:
{{"is_valid": true, "critique": "Patch is valid."}}
or
{{"is_valid": false, "critique": "<Reason why invalid>"}}
"""
        messages = [{"role": "user", "content": verification_prompt}]
        try:
            res = self.client.generate(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
                seed=seed
            )
            total_tokens = res.prompt_tokens + res.completion_tokens
            latency = res.latency_sec

            # Attempt to parse JSON response
            text = res.text.strip()
            # Clean possible markdown fences
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            try:
                data = json.loads(text)
                is_valid = bool(data.get("is_valid", True))
                critique = data.get("critique")
            except Exception:
                # If SLM output was not strict JSON, fallback to structural pass
                is_valid = True
                critique = None

            return {
                "is_valid": is_valid,
                "critique": critique if not is_valid else None,
                "tokens": total_tokens,
                "latency": latency
            }
        except Exception as e:
            logger.warning(f"QA Verifier SLM call failed: {e}. Falling back to structural validation pass.")
            return {
                "is_valid": True,
                "critique": None,
                "tokens": 0,
                "latency": 0.0
            }
