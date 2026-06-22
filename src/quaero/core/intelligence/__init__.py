from quaero.core.intelligence.base import EmbeddingEngine, InferenceEngine

from quaero.core.intelligence.providers import (
    OllamaProvider,
)

PROVIDER_REGISTRY = {
    "Ollama": OllamaProvider(),
}

def get_embedding_engine(name: str):
    engine = PROVIDER_REGISTRY.get(name)
    if not engine or not isinstance(engine, EmbeddingEngine):
        raise ValueError(f"Provider '{name}' does not support vector embeddings.")
        
    if not engine.is_functional():
        raise RuntimeError(
            f"⚠️ Provider '{name}' is configured but not functional.\n"
            f"If you are switching drivers, please run: pip install quaero[{name.lower()}]"
        )
    return engine

def get_inference_engine(name: str):
    engine = PROVIDER_REGISTRY.get(name)
    if not engine or not isinstance(engine, InferenceEngine):
        raise ValueError(f"Provider '{name}' does not support text inference.")
        
    if not engine.is_functional():
        raise RuntimeError(
            f"⚠️ Provider '{name}' is configured but not functional.\n"
            f"Ensure your local background service is running."
        )
    return engine