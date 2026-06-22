"""
Configuration file for the Quaero RAG system.
"""

import platformdirs
from pathlib import Path
from quaero.core import utils

# ==================================================
# Defaults
# ================================================== 

# Provider & Inference Configuration
DEFAULT_INFERENCE_PROVIDER = "Ollama"          
DEFAULT_INFERENCE_MODEL = "gemma4:e4b"

DEFAULT_EMBEDDING_PROVIDER = "Ollama"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"
DEFAULT_EMBEDDING_DIMENSIONS = 768

# Prompt template for RAG queries
DEFAULT_PROMPT_TEMPLATE = """
You are a precise information retrieval assistant attached to a local filesystem utility. Your core task is to answer the user's question by prioritizing the provided context snippets, while clearly distinguishing between ground-truth retrieved data and your own parametric knowledge.

Follow these strict output formatting rules:

1. IF THE INFORMATION IS FOUND IN THE RETRIEVED CONTEXT:
   - Start your response exactly with a section titled: "### 📄 Retrieved from Local Context"
   - Under this section, provide a precise, direct answer using ONLY the facts explicitly mentioned in the context.
   - If you have additional insights, external context, or advanced commentary to add from your own knowledge base, place it in a separate section titled: "### 💡 Supplemental Model Insights". Never mix your own assertions into the retrieved facts section.

2. IF THE QUESTION IS GENERAL OR NOT FOUND IN THE RETRIEVED CONTEXT:
   - Do NOT include a "Retrieved from Local Context" section.
   - Answer the question directly using your internal knowledge.
   - Start your response exactly with a brief disclaimer and then continue with section titled: "### 🌐 General Knowledge Response" (so the user knows no local files matched this query).

---
RETRIEVED CONTEXT CHUNKS:
{context}
---

USER QUESTION: {question}

FINAL ANSWER:
"""

# RAG Database Provider
DEFAULT_DATABASE_PROVIDER = "LanceDB"

# RAG Hyperparameters
DEFAULT_TOP_K = 7                           
DEFAULT_SCORE_THRESHOLD = 0.4         

# Text chunking configuration
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 80

# Ingestion pipeline
DEFAULT_BATCH_SIZE = 1000

# ==================================================
# Configuration Management & Paths
# ==================================================

# App Directories (Using Pythonic Path Division)
QUAERO_BASE_DIR = Path(platformdirs.user_data_dir(appname="quaero", appauthor="ADPer"))
QUAERO_CONFIG_DIR = Path(platformdirs.user_config_dir(appname="quaero", appauthor="ADPer"))

QUAERO_CONFIG_PATH = QUAERO_CONFIG_DIR / "config.toml"
QUAERO_DATABASE_PATH = QUAERO_BASE_DIR / "database"

# Initialize our defaults mapping
CONFIGS = {
    "INFERENCE_PROVIDER": DEFAULT_INFERENCE_PROVIDER,   
    "INFERENCE_MODEL": DEFAULT_INFERENCE_MODEL,
    "EMBEDDING_PROVIDER": DEFAULT_EMBEDDING_PROVIDER,
    "EMBEDDING_MODEL": DEFAULT_EMBEDDING_MODEL,
    "EMBEDDING_DIMENSIONS": DEFAULT_EMBEDDING_DIMENSIONS,
    "PROMPT_TEMPLATE": DEFAULT_PROMPT_TEMPLATE,
    "DATABASE_PROVIDER": DEFAULT_DATABASE_PROVIDER,
    "TOP_K": DEFAULT_TOP_K,
    "SCORE_THRESHOLD": DEFAULT_SCORE_THRESHOLD,
    "CHUNK_SIZE": DEFAULT_CHUNK_SIZE,
    "CHUNK_OVERLAP": DEFAULT_CHUNK_OVERLAP,
    "BATCH_SIZE": DEFAULT_BATCH_SIZE,
}

# Load existing user configs and cleanly merge them into the defaults
user_configs = utils.read_config(QUAERO_CONFIG_PATH)
if user_configs:
    CONFIGS.update(user_configs)

# Final exported intelligence variables (Safe to import elsewhere)
QUAERO_INFERENCE_PROVIDER = CONFIGS["INFERENCE_PROVIDER"]
QUAERO_INFERENCE_MODEL = CONFIGS["INFERENCE_MODEL"]

QUAERO_EMBEDDING_PROVIDER = CONFIGS["EMBEDDING_PROVIDER"]
QUAERO_EMBEDDING_MODEL = CONFIGS["EMBEDDING_MODEL"]
QUAERO_EMBEDDING_DIMENSIONS = CONFIGS["EMBEDDING_DIMENSIONS"]

QUAERO_PROMPT_TEMPLATE = CONFIGS["PROMPT_TEMPLATE"]

QUAERO_TOP_K = CONFIGS["TOP_K"]
QUAERO_SCORE_THRESHOLD = CONFIGS["SCORE_THRESHOLD"]

QUAERO_CHUNK_SIZE = CONFIGS["CHUNK_SIZE"]
QUAERO_CHUNK_OVERLAP = CONFIGS["CHUNK_OVERLAP"]
QUAERO_BATCH_SIZE = CONFIGS["BATCH_SIZE"]

# ==================================================
# App Directory Management
# ==================================================

def ensure_directories():
    """
    Ensures the necessary application directories exist, creates them if they don't.
    """
    directories = [
        ("BASE", QUAERO_BASE_DIR),
        ("CONFIGS", QUAERO_CONFIG_DIR),
        ("DATABASE", QUAERO_DATABASE_PATH), # <- LanceDB needs this to exist!
    ]
    
    for name, path in directories:
        try:
            # 'path' is already a pathlib.Path object, no need to wrap it again
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ {name} directory: {path}")
        except PermissionError:
            raise PermissionError(f"Permission denied: Cannot create {name} directory at {path}")
        except OSError as e:
            raise OSError(f"Failed to create {name} directory at {path}: {e}")