"""
Provider-Agnostic LLM Client Abstraction for Baseline Execution Harness.
Supports Anthropic (Claude 3.5 Sonnet) and OpenAI-compatible endpoints
(Groq, NVIDIA NIM, OpenAI, Ollama, etc.).

Features:
- Unified `generate(prompt, **kwargs)` interface returning `GenerationResult`.
- Configurable provider selection via `BASELINE_PROVIDER` (defaults to 'anthropic').
- Configurable rate-limiting throttle (default: 25 RPM for free-tier ceilings).
- Exponential backoff with jitter on HTTP 429 rate-limits and 5xx server errors.
- Respects `Retry-After` header when provided by endpoint.
- Zero extra SDK dependencies for OpenAI-compatible endpoints (uses `requests`).
"""
import os
import sys
import time
import json
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Attempt to load .env if python-dotenv is available
try:
    import dotenv
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _env_path = os.path.join(_proj_root, ".env")
    if os.path.exists(_env_path):
        dotenv.load_dotenv(_env_path)
    else:
        dotenv.load_dotenv()
except Exception:
    pass

logger = logging.getLogger("baseline_client")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GenerationResult:
    """Standardized response from any baseline LLM backend."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float
    provider: str
    model_name: str
    raw_response: Optional[Dict[str, Any]] = None

    # Allows legacy 4-tuple unpacking: patch_text, p_tokens, c_tokens, elapsed = result
    def __iter__(self):
        return iter((self.text, self.prompt_tokens, self.completion_tokens, self.latency_sec))

    def __getitem__(self, item):
        return (self.text, self.prompt_tokens, self.completion_tokens, self.latency_sec)[item]


class RPMThrottle:
    """
    Enforces a strict requests-per-minute ceiling via sliding interval delay.
    Prevents hitting free-tier RPM ceilings (e.g. Groq 30 RPM limit).
    """
    def __init__(self, max_rpm: int = 25):
        self.max_rpm = max_rpm
        self.min_interval = 60.0 / max(1, max_rpm)
        self.last_request_time: float = 0.0

    def wait_if_needed(self):
        if self.max_rpm <= 0:
            return
        now = time.time()
        elapsed_since_last = now - self.last_request_time
        if elapsed_since_last < self.min_interval:
            wait_time = self.min_interval - elapsed_since_last
            logger.info(
                f"[Throttle] Rate limit throttle ({self.max_rpm} RPM): "
                f"waiting {wait_time:.2f}s before next request..."
            )
            time.sleep(wait_time)
        self.last_request_time = time.time()


class BaseLLMBackend(ABC):
    """Abstract interface for baseline LLM backends."""
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs
    ) -> GenerationResult:
        """Executes LLM completion and returns GenerationResult."""
        pass


class AnthropicBackend(BaseLLMBackend):
    """
    Anthropic Claude 3.5 Sonnet backend.
    Preserves existing baseline call semantics untouched.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        model = model_name or os.environ.get("BASELINE_MODEL", "claude-3-5-sonnet-20241022")
        super().__init__(provider="anthropic", model_name=model)
        self.api_key = (
            api_key or
            os.environ.get("ANTHROPIC_API_KEY") or
            os.environ.get("BASELINE_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs
    ) -> GenerationResult:
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY or BASELINE_API_KEY in environment."
            )
        try:
            import anthropic
        except ImportError:
            raise ImportError("Package 'anthropic' is required for AnthropicBackend. Run `pip install anthropic`.")

        client = anthropic.Anthropic(api_key=self.api_key)
        start_time = time.time()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = time.time() - start_time
        patch_text = response.content[0].text
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        return GenerationResult(
            text=patch_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_sec=elapsed,
            provider="anthropic",
            model_name=self.model_name
        )


class OpenAICompatBackend(BaseLLMBackend):
    """
    OpenAI-compatible backend for Groq, NVIDIA NIM, OpenAI, Ollama, etc.
    Uses standard HTTP via `requests` to prevent third-party SDK version conflicts.
    Features:
    - Configurable base_url, model, and auth headers.
    - RPM throttling (default 25 RPM).
    - Exponential backoff with jitter on 429 and 5xx status codes.
    - Automatic `Retry-After` header inspection.
    """
    # Provider-specific default configurations
    PROVIDER_DEFAULTS = {
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "model": "openai/gpt-oss-120b",
            "suggested_models": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
            "default_rpm": 25
        },
        "nvidia": {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "model": "meta/llama-3.3-70b-instruct",
            "suggested_models": ["meta/llama-3.3-70b-instruct"],
            "default_rpm": 20
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-oss-120b",
            "suggested_models": ["openai/gpt-oss-120b"],
            "default_rpm": 25
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "suggested_models": ["gpt-4o", "gpt-4o-mini"],
            "default_rpm": 50
        }
    }

    def __init__(
        self,
        provider: str = "groq",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        max_rpm: Optional[int] = None,
        max_retries: int = 5,
        timeout_sec: float = 60.0
    ):
        defaults = self.PROVIDER_DEFAULTS.get(provider.lower(), {})
        resolved_model = (
            model_name or
            os.environ.get("BASELINE_MODEL") or
            defaults.get("model", "openai/gpt-oss-120b")
        )
        super().__init__(provider=provider.lower(), model_name=resolved_model)

        self.api_key = (
            api_key or
            os.environ.get("BASELINE_API_KEY") or
            os.environ.get("GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY")
        )

        resolved_base_url = (
            base_url or
            os.environ.get("BASELINE_BASE_URL") or
            defaults.get("base_url", "https://api.groq.com/openai/v1")
        )
        self.base_url = resolved_base_url.rstrip("/")

        # Throttle configuration
        env_rpm = os.environ.get("BASELINE_MAX_RPM")
        resolved_rpm = int(env_rpm) if env_rpm else (max_rpm or defaults.get("default_rpm", 25))
        self.throttle = RPMThrottle(max_rpm=resolved_rpm)

        self.max_retries = max_retries
        self.timeout_sec = timeout_sec

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs
    ) -> GenerationResult:
        if not self.api_key:
            raise ValueError(
                f"Missing API key for provider '{self.provider}'. "
                f"Set BASELINE_API_KEY in environment or .env."
            )

        key_str = self.api_key.strip().lower()
        if not key_str or "your_groq_api_key" in key_str or "your_api_key" in key_str:
            raise ValueError(
                f"BASELINE_API_KEY is set to placeholder '{self.api_key}'. "
                f"Please update .env with your real Groq API key."
            )

        import requests

        resolved_base_url = self.base_url
        extra_headers = {}
        if self.api_key.startswith("sk-or-"):
            if "groq.com" in resolved_base_url:
                resolved_base_url = "https://openrouter.ai/api/v1"
            extra_headers["HTTP-Referer"] = "https://github.com/MicroSWE-Baseline"
            extra_headers["X-Title"] = "MicroSWE-Baseline-Prototype"

        url = f"{resolved_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MicroSWE-Baseline-Prototype/1.0",
            **extra_headers
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Merge optional kwargs
        for k, v in kwargs.items():
            if k not in payload and k != "seed":
                payload[k] = v

        attempt = 0
        backoff_base = 2.0

        while attempt <= self.max_retries:
            # Enforce RPM throttle before each outbound request
            self.throttle.wait_if_needed()

            start_time = time.time()
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError(f"Empty choices returned from {self.provider}: {data}")

                    msg = choices[0].get("message", {})
                    patch_text = msg.get("content") or ""
                    if not patch_text and "reasoning" in msg:
                        patch_text = msg.get("reasoning") or ""

                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

                    return GenerationResult(
                        text=patch_text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_sec=elapsed,
                        provider=self.provider,
                        model_name=self.model_name,
                        raw_response=data
                    )

                # Rate Limit handling (429)
                elif response.status_code == 429:
                    attempt += 1
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = float(retry_after) + 0.5
                        except ValueError:
                            wait_time = backoff_base * (2 ** attempt) + random.uniform(0.1, 0.5)
                    else:
                        wait_time = backoff_base * (2 ** attempt) + random.uniform(0.1, 0.5)

                    logger.warning(
                        f"[RateLimit 429] Endpoint returned 429 (Too Many Requests). "
                        f"Backing off for {wait_time:.2f}s (attempt {attempt}/{self.max_retries})."
                    )
                    time.sleep(wait_time)
                    continue

                # Server errors (5xx)
                elif response.status_code in (500, 502, 503, 504):
                    attempt += 1
                    wait_time = backoff_base * (2 ** attempt) + random.uniform(0.1, 0.5)
                    logger.warning(
                        f"[ServerError {response.status_code}] Backend transient error: {response.text[:200]}. "
                        f"Retrying in {wait_time:.2f}s (attempt {attempt}/{self.max_retries})."
                    )
                    time.sleep(wait_time)
                    continue

                # Authentication / Bad Request / Client error
                else:
                    err_msg = f"HTTP {response.status_code} from {self.provider} ({url}): {response.text[:400]}"
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
                attempt += 1
                wait_time = backoff_base * (2 ** attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    f"[NetworkError] {str(net_err)}. Retrying in {wait_time:.2f}s "
                    f"(attempt {attempt}/{self.max_retries})."
                )
                time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to obtain response from {self.provider} ({self.model_name}) "
            f"after {self.max_retries} retries."
        )


def get_baseline_client(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_rpm: Optional[int] = None
) -> BaseLLMBackend:
    """
    Factory creating the configured LLM backend.
    Selection order:
    1. Explicit `provider` argument if supplied.
    2. `BASELINE_PROVIDER` environment variable.
    3. Defaults to 'anthropic' (guaranteeing zero regression for real baseline).
    """
    prov = (provider or os.environ.get("BASELINE_PROVIDER", "anthropic")).lower().strip()

    if prov == "anthropic":
        return AnthropicBackend(api_key=api_key, model_name=model_name)
    elif prov in ("groq", "openrouter", "nvidia", "openai", "openai_compat"):
        return OpenAICompatBackend(
            provider=prov,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            max_rpm=max_rpm
        )
    else:
        raise ValueError(
            f"Unknown BASELINE_PROVIDER '{prov}'. "
            f"Supported providers: 'anthropic', 'groq', 'openrouter', 'nvidia', 'openai', 'openai_compat'."
        )
