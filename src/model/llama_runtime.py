"""Singleton CPU-only runtime wrapper for llama.cpp / Qwen2.5-Coder-3B.

Ensures the 3B GGUF model is loaded exactly ONCE into system memory
and shared safely across all pipeline stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any

from src.config import AERSConfig, get_config
from src.schemas import estimate_token_count

logger = logging.getLogger(__name__)

# Optional llama_cpp import
try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    llama_cpp = None  # type: ignore[assignment]
    LLAMA_CPP_AVAILABLE = False


class LlamaRuntime:
    """Thread-safe singleton managing the loaded GGUF model."""

    _instance: LlamaRuntime | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, config: AERSConfig | None = None) -> None:
        self.config = config or get_config()
        self.model: Any = None
        self._inference_lock = threading.Lock()
        self._load_model()

    @classmethod
    def get_instance(cls, config: AERSConfig | None = None) -> LlamaRuntime:
        """Return or lazily initialize the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                del cls._instance.model
                cls._instance = None

    def _load_model(self) -> None:
        """Load GGUF weights into CPU memory."""
        if self.config.dry_run:
            logger.info("LlamaRuntime initialized in dry_run mode. Model weights skipped.")
            return

        if not LLAMA_CPP_AVAILABLE:
            logger.warning(
                "llama-cpp-python is not installed. LlamaRuntime operating in offline mode. "
                "Use FakeLLM or install llama-cpp-python to run inference."
            )
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            logger.warning(
                "Model file not found at %s. Inference will require model download or FakeLLM.",
                model_path,
            )
            return

        logger.info(
            "Loading Qwen2.5-Coder-3B GGUF from %s (threads=%d, ctx=%d, batch=%d, n_gpu_layers=0)",
            model_path,
            self.config.n_threads,
            self.config.n_ctx,
            self.config.n_batch,
        )

        self.model = llama_cpp.Llama(
            model_path=str(model_path),
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_batch=self.config.n_batch,
            n_gpu_layers=0,  # Strictly CPU-only
            seed=self.config.seed,
            use_mmap=self.config.use_mmap,
            use_mlock=self.config.use_mlock,
            chat_format="chatml",
            verbose=False,
        )
        logger.info("Successfully loaded GGUF model into CPU memory.")

    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self.model is not None

    def tokenize(self, text: str) -> list[int]:
        """Tokenize string using model tokenizer if loaded, else fallback."""
        if self.is_loaded():
            try:
                return self.model.tokenize(text.encode("utf-8"))
            except Exception as e:
                logger.warning("Tokenize failed on model: %s. Using fallback.", e)
        # Fallback approximation: list of dummy token IDs with length equal to estimate
        count = estimate_token_count(text)
        return list(range(count))

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using the loaded tokenizer, or fallback estimator."""
        if self.is_loaded():
            try:
                return len(self.model.tokenize(text.encode("utf-8")))
            except Exception:
                pass
        return estimate_token_count(text)
