"""
Embedding utilities for Quaero.
"""

from typing import List

from quaero.core.config import (
    QUAERO_EMBEDDING_MODEL,
    QUAERO_EMBEDDING_PROVIDER,
    QUAERO_EMBEDDING_DIMENSIONS,
)
from quaero.core.intelligence import get_embedding_engine

# Initializing the embedding engine for the configured provider
EMBEDDING_ENGINE = get_embedding_engine(QUAERO_EMBEDDING_PROVIDER)


def get_embedding(chunks: List[str]) -> List[List[float]]:
    """
    Returns vector embeddings for a list of text chunks based on the configurations.
    """
    try:
        embeddings = EMBEDDING_ENGINE.get_embedding(chunks, QUAERO_EMBEDDING_MODEL, QUAERO_EMBEDDING_DIMENSIONS)
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        raise

def embed_query(query: str) -> List[float]:
    """
    Returns the vector embedding for a single query string.
    """
    try:
        embedding = EMBEDDING_ENGINE.get_embedding([query], QUAERO_EMBEDDING_MODEL, QUAERO_EMBEDDING_DIMENSIONS)[0]
        return embedding
    except Exception as e:
        print(f"Error embedding query: {e}")
        raise