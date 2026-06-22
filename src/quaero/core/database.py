"""
Database management and operations layer for the Quaero RAG engine.
"""

import lancedb
from typing import List, Optional, Dict, Any
from pathlib import Path

from quaero.core.config import QUAERO_DATABASE_PATH, QUAERO_TOP_K
from quaero.core.models import FileRecord, ChunkRecord


class QuaeroDatabase:
    """
    Singleton-style manager for LanceDB operations. 
    Handles table initialization, relational cascade emulation, and vector search.
    """

    def __init__(self, db_path: Path  = QUAERO_DATABASE_PATH):
        # Initializes the memory-mapped connection. 
        # If the directory doesn't exist, LanceDB creates the .lance structure natively.
        self.db = lancedb.connect(str(db_path))
        self._initialize_tables()

    def _initialize_tables(self):
        """Ensures the core schema tables exist on disk."""
        existing_tables = self.db.table_names()
        
        if "files" not in existing_tables:
            self.db.create_table("files", schema=FileRecord)
            
        if "chunks" not in existing_tables:
            self.db.create_table("chunks", schema=ChunkRecord)

    # ==========================================
    # FILE MANAGEMENT (The Relational Logic)
    # ==========================================

    def upsert_file_record(self, record: FileRecord):
        """
        Inserts a new file record, or overwrites an existing one.
        """
        table = self.db.open_table("files")
        
        # Emulate an UPSERT by deleting any existing record with this ID first
        table.delete(f"file_id = '{record.file_id}'")
        table.add([record.model_dump()])

    def get_file_record(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves file metadata by its hash ID."""
        table = self.db.open_table("files")
        results = table.search().where(f"file_id = '{file_id}'").limit(1).to_list()
        return results[0] if results else None

    def remove_file(self, file_id: str):
        """
        Application-layer ON DELETE CASCADE. 
        Wipes the file tracking record AND purges all associated vector chunks.
        """
        files_table = self.db.open_table("files")
        chunks_table = self.db.open_table("chunks")
        
        # Purge the vectors first
        chunks_table.delete(f"file_id = '{file_id}'")
        # Purge the tracking metadata
        files_table.delete(f"file_id = '{file_id}'")

    # ==========================================
    # CHUNK & VECTOR MANAGEMENT
    # ==========================================

    def insert_chunks(self, chunks: List[ChunkRecord]):
        """
        Batches an array of ChunkRecords into the vector index.
        """
        if not chunks:
            return
            
        table = self.db.open_table("chunks")
        
        # LanceDB's .add() expects a list of dictionaries matching the schema
        table.add([chunk.model_dump() for chunk in chunks])

    def search_chunks(self, query_vector: List[float], top_k: int = QUAERO_TOP_K) -> List[Dict[str, Any]]:
        """
        Performs a high-speed L2/Cosine vector similarity search against the embedded chunks.
        """
        table = self.db.open_table("chunks")
        
        # The true power of LanceDB: chaining vector search with limits
        results = (
            table.search(query_vector)
            .distance_type("cosine")  # type: ignore
            .limit(top_k)
            .to_list()
        )
        return results