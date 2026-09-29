"""Base agent abstraction for AERS-SLM sequential pipeline stages.

Enforces zero conversation history retention between stages: agents receive
ONLY the bounded EvidenceBaton and the compiled context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Generic, TypeVar

from src.model.inference import InferenceEngine
from src.schemas import EvidenceBaton

logger = logging.getLogger(__name__)

T_Out = TypeVar("T_Out")


class BaseAgent(ABC, Generic[T_Out]):
    """Base class for all specialized AERS-SLM agents."""

    def __init__(
        self,
        engine: InferenceEngine,
        system_prompt: str,
        stage_name: str = "base_agent",
    ) -> None:
        self.engine = engine
        self.system_prompt = system_prompt
        self.stage_name = stage_name

    @abstractmethod
    def run(self, baton: EvidenceBaton, context: str, **kwargs: Any) -> T_Out:
        """Execute the agent stage given the baton and context.
        
        Must never retain or reference conversational history from prior calls.
        """
        raise NotImplementedError
