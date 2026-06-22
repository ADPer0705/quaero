"""
Quaero Core Engine
==================
The underlying systems, database operations, and intelligence routing for Quaero.
"""

# Configuration & Paths
from .config import (
    QUAERO_BASE_DIR,
    QUAERO_DATABASE_PATH,
    QUAERO_CONFIG_PATH,
    ensure_directories
)

# Database ops
from .database import QuaeroDatabase

# Pipeline actions
from .ingest import ingest_pipeline
from .query import query_rag

# Intelligence SDKs
from .intelligence import get_inference_engine, get_embedding_engine

# Versioning
from .utils import get_app_version

__all__ = [
    "QUAERO_BASE_DIR",
    "QUAERO_DATABASE_PATH",
    "QUAERO_CONFIG_PATH",
    "ensure_directories",
    "QuaeroDatabase",
    "ingest_pipeline",
    "query_rag",
    "get_inference_engine",
    "get_embedding_engine",
    "get_app_version"
]