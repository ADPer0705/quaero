"""
Abstract base definitions for intelligence providers in Quaero.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

class BaseProvider(ABC):
    """Core interface that all AI providers must implement."""
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the unique identifier for the provider (e.g., 'ollama')."""
        pass

    @abstractmethod
    def is_functional(self) -> bool:
        """Verifies if the local daemon/API endpoint is up and responding."""
        pass


class EmbeddingEngine(BaseProvider, ABC):
    """Interface for providers capable of generating vector embeddings."""

    @abstractmethod
    def get_embedding(self, chunks: List[str], model: str) -> List[List[float]]:
        """Generates vector embeddings for a list of text chunks."""
        pass


class InferenceEngine(BaseProvider, ABC):
    """Interface for providers capable of executing LLM generation tasks."""

    @abstractmethod
    def generate(self, prompt: str, model: str, system_prompt: Optional[str] = None) -> str:
        """Executes a standard non-streaming text generation request."""
        pass