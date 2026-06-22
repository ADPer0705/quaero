# 🔬 Quaero

<div align="center">
  <p><strong>High-Performance, Local-First RAG Document Assistant</strong></p>
  <p><em>Transform your local documents into an intelligent, queryable knowledge base—without the bloat.</em></p>
  
  [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  [![LanceDB](https://img.shields.io/badge/LanceDB-Vector_Store-orange.svg)](https://lancedb.com/)
  [![Ollama](https://img.shields.io/badge/Ollama-Local_Intelligence-purple.svg)](https://ollama.ai/)
</div>

---

## 🎯 Overview

**Quaero** is a streamlined, local-first Retrieval-Augmented Generation (RAG) engine. Built for developers, researchers, and engineers, it completely bypasses heavy frameworks like LangChain in favor of a custom, memory-flat ingestion pipeline and blazing-fast vector search via LanceDB.

Your data never leaves your machine. 

## ✨ The Engineering Edge

- **Tiered Ingestion Router:** Automatically routes files to the most efficient parser (e.g., C-bound `PyMuPDF` for PDFs, native `python-docx` for Word, and raw streaming for code/text) while bouncing binary executables at the door.
- **Memory-Flat Processing:** Reads and hashes massive files (like 1,000-page textbooks) using lazy generators, keeping your RAM usage practically at zero during ingestion.
- **State Reconciliation:** Native `sync` tracking detects when you modify or delete a physical file and automatically purges or updates the orphaned vectors via relational metadata.
- **Zero-Config Vector Search:** Powered by LanceDB's PyArrow backend for native, sub-millisecond Cosine distance retrieval.

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Ollama** installed and running locally.

### 1. Installation
Install directly via pip (or `pipx` for isolated environments):
```bash
pip install quaero
```

### 2. Initial Setup
Run the interactive wizard to configure your models and chunk sizes:

```bash
quaero setup
```

(We recommend embeddinggemma for embeddings and a fast, instruction-tuned model like gemma or llama3 for inference).

### 3. Build Your Knowledge Base
Point Quaero at a single file or an entire directory. It will recursively crawl and index supported formats.

```Bash
quaero ingest /path/to/your/documents/
```

### 4. Start Querying
Launch the interactive terminal UI to chat with your documents:

```Bash
quaero chat
```

Or execute a single-shot query:

```Bash
quaero chat "What are the main persistence mechanisms described in the malware textbook?"
```

## 💻 CLI Command Reference

Quaero features a modern, Rich-powered CLI.
- quaero status - View database health and vector counts.
- quaero ingest <path> - Ingest a file or directory.
- quaero sync - Reconcile the vector database with your physical filesystem (purges orphans, updates modifications).
- quaero config show - Display active thresholds, models, and chunk parameters.
- quaero config set <KEY> <VALUE> - Tune the engine on the fly (e.g., quaero config set score_threshold 0.6).
- quaero db reset - Nuke the database and start fresh.

## 🏗️ Architecture

graph TD
    A[Local Filesystem] -->|quaero sync / ingest| B[Tiered Extraction Router]
    B --> C[Memory-Flat Text Splitter]
    C --> D[Ollama Embedding Engine]
    D --> E[(LanceDB Vector Store)]
    
    F[User Query] --> G[Cosine Similarity Search]
    G --> E
    E --> H[Context Assembly]
    H --> I[Ollama Inference]
    I --> J[Grounded Terminal Response]