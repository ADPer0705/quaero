"""
Ollama service provider implementation utilizing on-demand runtime loading.
"""

from typing import List, Optional
from quaero.core.intelligence.base import EmbeddingEngine, InferenceEngine
from quaero.core.utils import install_provider_sdk, is_library_installed


class OllamaProvider(EmbeddingEngine, InferenceEngine):

    def __init__(self):
        # Cache the operational state so we don't spam the daemon during batch operations
        self._is_ready: Optional[bool] = None

    @property
    def provider_name(self) -> str:
        return "Ollama"

    def _ensure_driver(self) -> bool:
        """Internal helper to verify or inject the required SDK."""
        return install_provider_sdk(self.provider_name, "ollama")

    def is_functional(self) -> bool:
        """
        Checks if the driver is loaded and the local daemon is responsive.
        Result is cached to ensure high-performance execution inside tight loops.
        """
        if self._is_ready is not None:
            return self._is_ready

        # 1. Handle missing driver installation inline
        if not is_library_installed("ollama"):
            if not self._ensure_driver():
                self._is_ready = False
                return False

        # 2. Verify connection viability with the daemon
        try:
            import ollama
            ollama.list()
            self._is_ready = True
            return True
        except Exception:
            self._is_ready = False
            return False

    def get_embedding(self, chunks: List[str], model: str, expected_dimensions: Optional[int] = None) -> List[List[float]]:
        if not self.is_functional():
            raise RuntimeError("Ollama service is unavailable or driver installation failed.")
            
        import ollama
        
        try:
            # Enforce modern SDK usage exclusively
            response = ollama.embed(
                model=model,
                input=chunks,
                dimensions=expected_dimensions
            )
            # .embed returns a 2D array matrix under 'embeddings'
            embeddings = response["embeddings"]
            
        except AttributeError:
            # The client SDK is too old and lacks the .embed() method
            raise RuntimeError(
                "🚨 Outdated Driver: Your Ollama Python SDK is missing the modern '.embed' interface. "
                "Please upgrade it by running 'pip install --upgrade ollama'."
            )
        except KeyError:
            # The daemon responded, but the JSON payload was malformed or missing the embeddings key
            raise RuntimeError("🚨 API Error: The Ollama daemon returned an unexpected or malformed response payload.")
        except Exception as e:
            # Catch network timeouts, connection drops, or model loading failures
            raise RuntimeError(f"🚨 Ollama embedding generation failed: {str(e)}")
            
        # Defensive System Guard: Strict Structural Verification
        if expected_dimensions and len(embeddings[0]) != expected_dimensions:
            raise ValueError(
                f"🚨 Vector shape mismatch! System configured for {expected_dimensions} dimensions "
                f"using model '{model}', but received a vector array of length {len(embeddings[0])}."
            )
            
        return embeddings

    def generate(self, prompt: str, model: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_functional():
            raise RuntimeError("Ollama service is unavailable or driver installation failed.")
            
        import ollama
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]