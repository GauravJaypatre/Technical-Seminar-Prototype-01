"""Configuration management for AERS-SLM.

All tunables in one place, overridable via environment variables (AERS_*)
or YAML configuration files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psutil
from pydantic import BaseModel, Field
import yaml


def default_cpu_threads() -> int:
    """Return optimal CPU inference thread count.
    
    Empirically benchmarked on Intel 12th-Gen Core i5-1235U (2 P-cores + 8 E-cores):
    - 4 threads:  9.33 tok/s
    - 6 threads: 10.10 tok/s (optimal throughput without P/E core thread thrashing)
    - 8 threads:  9.54 tok/s
    - 10 threads: 9.01 tok/s
    - 12 threads: 8.02 tok/s (over 20% slower due to thread contention)
    """
    return 6


class AERSConfig(BaseModel):
    """Central configuration for AERS-SLM runtime and pipeline."""

    # Model & Runtime
    model_path: str = Field(
        default="models/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        description="Path to Qwen2.5-Coder-3B-Instruct GGUF weight file."
    )
    n_ctx: int = Field(default=8192, description="Context window size (tokens).")
    n_threads: int = Field(
        default_factory=default_cpu_threads,
        description="Number of CPU threads for inference (default: physical cores)."
    )
    n_batch: int = Field(default=512, description="Batch size for prompt processing.")
    temperature: float = Field(default=0.2, description="Sampling temperature.")
    top_p: float = Field(default=0.95, description="Nucleus sampling top-p.")
    seed: int = Field(default=42, description="Fixed RNG seed for deterministic generation.")
    use_mmap: bool = Field(default=True, description="Memory-map model weights.")
    use_mlock: bool = Field(default=False, description="Lock model in RAM (CPU only).")
    n_gpu_layers: int = Field(default=0, description="CPU-only execution (strictly 0).")

    # Per-stage token ceilings
    max_tokens_profiler: int = Field(default=500, description="Max tokens for Stage 1.")
    max_tokens_localizer: int = Field(default=600, description="Max tokens for Stage 3.")
    max_tokens_hypothesis: int = Field(default=600, description="Max tokens for Stage 4.")
    max_tokens_patch: int = Field(default=1024, description="Max tokens for Stage 6.")

    # Budgets & Limits
    context_budget: int = Field(default=4000, description="Max tokens for compiled context.")
    baton_max_tokens: int = Field(default=1200, description="Hard cap for EvidenceBaton.")
    initial_baton_max_tokens: int = Field(default=500, description="Budget for initial B0 baton.")
    max_retries: int = Field(default=2, description="Max patch retry attempts.")

    # Logging & Run Mode
    log_dir: str = Field(default="logs", description="Directory for structured JSONL logs.")
    dry_run: bool = Field(default=False, description="Dry-run / mock mode without loading GGUF.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> AERSConfig:
        """Load configuration from a YAML file, merging with env vars."""
        file_path = Path(path)
        data: dict[str, Any] = {}
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    data = loaded
        return cls.from_dict_and_env(data)

    @classmethod
    def from_env(cls) -> AERSConfig:
        """Load configuration from environment variables (prefixed with AERS_)."""
        return cls.from_dict_and_env({})

    @classmethod
    def from_dict_and_env(cls, base: dict[str, Any]) -> AERSConfig:
        """Build config from dictionary overlayed with AERS_* environment variables."""
        merged = dict(base)
        for key in cls.model_fields.keys():
            env_var = f"AERS_{key.upper()}"
            if env_var in os.environ:
                val = os.environ[env_var]
                field_info = cls.model_fields[key]
                # Cast to field type
                if field_info.annotation is int:
                    merged[key] = int(val)
                elif field_info.annotation is float:
                    merged[key] = float(val)
                elif field_info.annotation is bool:
                    merged[key] = val.lower() in ("true", "1", "yes")
                else:
                    merged[key] = val
        return cls(**merged)


# Global default configuration instance
_GLOBAL_CONFIG: AERSConfig | None = None


def get_config(reload: bool = False) -> AERSConfig:
    """Return the global configuration instance."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None or reload:
        # Check if AERS_CONFIG_PATH is set
        config_path = os.environ.get("AERS_CONFIG_PATH", "config.yaml")
        if os.path.exists(config_path):
            _GLOBAL_CONFIG = AERSConfig.from_yaml(config_path)
        else:
            _GLOBAL_CONFIG = AERSConfig.from_env()
    return _GLOBAL_CONFIG
