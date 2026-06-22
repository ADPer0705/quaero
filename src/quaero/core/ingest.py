'''
Ingest Module for Quaero
'''
#TODO: Improve performance

import time
import hashlib
from typing import Iterator, List
from itertools import islice
from pathlib import Path

from quaero.core.models import FileRecord, ChunkRecord
from quaero.core.database import QuaeroDatabase
from quaero.core.embedding import get_embedding
from quaero.core.config import (
    QUAERO_CHUNK_SIZE,
    QUAERO_CHUNK_OVERLAP,
    QUAERO_BATCH_SIZE,    
)

# ==================================================
#  Helpers & Extraction Router
# ==================================================

# Hashing
def compute_content_hash(file_path: Path) -> str:
    """Computes a SHA-256 hash of the file content in flat 8KB blocks."""
    hasher = hashlib.sha256()
    with file_path.open('rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_path_hash(file_path: Path) -> str:
    """Computes a deterministic hash from the absolute path to act as our primary key."""
    return hashlib.sha256(str(file_path).encode('utf-8')).hexdigest()

# Chunking
def extract_text_stream(file_path: Path) -> Iterator[str]:
    """
    Smart router that identifies file type and streams text line-by-line.
    Keeps memory flat by using lazy extraction where possible.
    """
    ext = file_path.suffix.lower()

    # Tier 0: The Binary Wall (Bounce media and executables instantly)
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.mp4', '.mp3', '.wav', 
        '.exe', '.dll', '.so', '.zip', '.tar', '.gz', '.iso', '.bin'
    }
    if ext in binary_extensions:
        print(f"⏭️  Skipping binary file: {file_path.name}")
        yield ""
        return

    try:
        # Tier 1a: PDF (The Speed Demon)
        if ext == '.pdf':
            import pymupdf  # Modern import standard
            with pymupdf.open(file_path) as doc:
                for page in doc:
                    # Cast to str to satisfy Pylance's strict Union type checking
                    text = str(page.get_text())
                    for line in text.split('\n'):
                        if line.strip():
                            yield line

        # Tier 1b: Word Documents
        elif ext in ['.docx', '.doc']:
            doc = docx.Document(file_path)  # type: ignore
            
            for para in doc.paragraphs:
                if para.text.strip():
                    yield para.text

        # Tier 2: The Sledgehammer (Complex presentations & web formats)
        elif ext in ['.pptx', '.html', '.eml', '.epub']:
            from unstructured.partition.auto import partition
            elements = partition(filename=str(file_path))
            for element in elements:
                text = str(element).strip()
                if text:
                    yield text

        # Tier 3: The Universal Catch-All (md, txt, csv, py, rs, nix, etc.)
        else:
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip():
                        yield line

    except Exception as e:
        print(f"⚠️ Failed to extract text from {file_path.name}: {e}")
        yield ""


def structural_text_splitter(line_iterator: Iterator[str], chunk_size: int = int(QUAERO_CHUNK_SIZE), overlap: int = QUAERO_CHUNK_OVERLAP) -> Iterator[str]:
    buffer = ""
    for line in line_iterator:
        buffer += line
        while len(buffer) >= chunk_size:
            split_idx = buffer.rfind(" ", 0, chunk_size)
            if split_idx == -1:
                split_idx = chunk_size
            chunk = buffer[:split_idx].strip()
            if chunk:
                yield chunk
            buffer = buffer[split_idx - overlap:]
    if buffer.strip():
        yield buffer.strip()

def chunk_batch_generator(chunk_iterator: Iterator[str], batch_size: int = QUAERO_BATCH_SIZE) -> Iterator[List[str]]:
    while True:
        batch = list(islice(chunk_iterator, batch_size))
        if not batch:
            break
        yield batch

# ==================================================
#  Main Ingestion Logic
# ==================================================

def ingest_pipeline(file_path: Path, db: QuaeroDatabase) -> bool:
    """
    Orchestrates the end-to-end ingestion of a local file into the LanceDB vector store.
    """
    abs_path = file_path.resolve()
    
    if not abs_path.exists():
        raise FileNotFoundError(f"Cannot ingest missing file: {abs_path}")

    file_stat = abs_path.stat()
    current_mtime = file_stat.st_mtime
    path_id = compute_path_hash(abs_path)
    current_content_hash = compute_content_hash(abs_path)

    # 1. Update Detection Logic
    existing_record = db.get_file_record(path_id)
    
    if existing_record:
        if existing_record.get("content_hash") == current_content_hash:
            return False 
        db.remove_file(path_id)

    # 2. Register Metadata
    file_record = FileRecord(
        file_id=path_id,
        content_hash=current_content_hash,
        file_name=abs_path.name,
        absolute_path=str(abs_path),
        file_size_bytes=file_stat.st_size,
        last_modified=current_mtime,
        indexed_at=time.time()
    )
    db.upsert_file_record(file_record)

    # 3. Stream, Split, and Batch (Using our new router!)
    lines = extract_text_stream(abs_path)
    semantic_chunks = structural_text_splitter(lines)
    batches = chunk_batch_generator(semantic_chunks)

    # 4. The Embedding & Writing Loop
    global_chunk_index = 0
    
    for batch in batches:
        vectors = get_embedding(chunks=batch)
        
        chunk_records = []
        for i, (text, vector) in enumerate(zip(batch, vectors)):
            record = ChunkRecord(
                chunk_id=f"{path_id}_c{global_chunk_index + i}",
                file_id=path_id,
                text_content=text,
                chunk_index=global_chunk_index + i,
                vector=vector
            )
            chunk_records.append(record)
            
        db.insert_chunks(chunk_records)
        global_chunk_index += len(batch)
        
    return True