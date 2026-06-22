"""
Query utilities for Quaero.
"""

import time
from typing import Optional

from quaero.core.config import (
    QUAERO_INFERENCE_MODEL,
    QUAERO_INFERENCE_PROVIDER,
    QUAERO_PROMPT_TEMPLATE,
    QUAERO_DATABASE_PATH,
    QUAERO_TOP_K,
    QUAERO_SCORE_THRESHOLD,
)
from quaero.core.database import QuaeroDatabase
from quaero.core.intelligence import get_inference_engine
from quaero.core.embedding import embed_query

# ==================================================
# Initialization
# ==================================================

# Database
db = QuaeroDatabase(QUAERO_DATABASE_PATH)

# Intelligence Engines
# TODO: Organize inference logic separately like embedding logic
# TODO: Figure out proper utilization of System prompt
INFERENCE_ENGINE = get_inference_engine(QUAERO_INFERENCE_PROVIDER)

def get_response(prompt: str) -> str:
    """
    Generates a response using explicit System and User roles to prevent prompt injection 
    and keep the LLM strictly grounded to the provided context.
    """
    try:
        response = INFERENCE_ENGINE.generate(
            prompt=prompt, 
            model=QUAERO_INFERENCE_MODEL,
        )
        return response
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        raise

# ==================================================
# RAG Query Logic
# ==================================================

def query_rag(
        query_text: str, 
        k: int = QUAERO_TOP_K, 
        threshold: float = QUAERO_SCORE_THRESHOLD, 
    ) -> Optional[dict]:
    """
    Query the RAG system using the provided text.
    """
    start_time = time.time()
    
    print(f"🔍 Query: {query_text}")
    print(f"📊 Retrieving top {k} documents with distance threshold {threshold}")
    
    if not query_text.strip():
        print("❌ Error: Query cannot be empty")
        return None
        
    # 1. Embed the Question
    try:
        query_vector = embed_query(query_text)
    except Exception as e:
        print(f"❌ Error embedding query: {e}")
        return None

    # 2. Vector Search
    raw_sources = []
    try:
        results = db.search_chunks(query_vector, top_k=k)
        
        # Filter by distance (Lower is better in L2 space)
        filtered_results = [row for row in results if row["_distance"] <= threshold]
        
        if not filtered_results:
            print(f"❌ No documents found within distance threshold {threshold}")
        
        print(f"📄 Found {len(filtered_results)} relevant chunks")
        
        for i, chunk in enumerate(filtered_results):
            distance = chunk["_distance"]
            file_record = db.get_file_record(chunk["file_id"])
            source = file_record["file_name"] if file_record else f"Unknown ({chunk['file_id']})"
            
            raw_sources.append(source)
            print(f"   {i+1}. Distance: {distance:.3f} | Source: {source}")
            
    except Exception as e:
        print(f"❌ Error during similarity search: {e}")
        return None

    # Deduplicate sources while preserving ranking order
    unique_sources = list(dict.fromkeys(raw_sources))

    # 3. Build the Context Payload
    context_text = "\n\n---\n\n".join([chunk["text_content"] for chunk in filtered_results])
    
    # Isolate the roles for the LLM
    system_context = (
        "You are a precise, technical assistant. Answer the user's question using ONLY the provided context below. "
        "If the answer is not contained in the context, explicitly state that you do not know. Do not hallucinate.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    # 4. Generate Answer
    prompt = QUAERO_PROMPT_TEMPLATE.format(context=system_context, question=query_text)

    try:
        print("🤖 Generating response...")
        response_text = get_response(prompt=prompt)
    except Exception:
        print("💡 Check if the backend daemon is running and the model is available.")
        return None

    response_time = time.time() - start_time
    
    response_data = {
        "answer": response_text,
        "sources": unique_sources,
        "num_sources": len(unique_sources),
        "response_time": response_time,
        "query": query_text
    }
    
    # 5. Render Final CLI Output
    print("\n" + "="*50)
    print("🤖 Answer:")
    print(response_text)
    print("\n" + "-"*30)
    
    print(f"📚 Unique Sources ({len(unique_sources)}):")
    for i, source in enumerate(unique_sources, 1):
        print(f"   {i}. {source}")
    
    # Fixed indentation!
    print(f"\n⏱️  Response time: {response_time:.2f}s")
    print("="*50)
    
    return response_data