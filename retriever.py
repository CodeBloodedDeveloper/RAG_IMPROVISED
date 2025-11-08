# retriever.py - REWRITTEN FOR PINECONE

import os
import asyncio
from pinecone import Pinecone
from typing import Optional, Tuple, List, Dict, Any

from config import PINEONE_API_KEY, PINEONE_ENVIRONMENT, AGENT_CONFIG, VECTOR_INDEX_NAME
from embeddings import embed_documents

# --- Global Cache for Pinecone Connection and Index ---
_PINEONE_CLIENT = None
_PINEONE_INDEX = None 

def _get_pinecone_index(agent_name: str):
    """
    Initializes and returns a cached Pinecone Index object for runtime queries.
    """
    global _PINEONE_CLIENT, _PINEONE_INDEX

    if not PINEONE_API_KEY or not PINEONE_ENVIRONMENT:
        print("⚠️ Pinecone API key or Environment not configured. Retrieval skipped.")
        return None

    if _PINEONE_CLIENT is None:
        try:
            _PONEONE_CLIENT = Pinecone(api_key=PINEONE_API_KEY, environment=PINEONE_ENVIRONMENT)
            _PINEONE_INDEX = _PONEONE_CLIENT.Index(VECTOR_INDEX_NAME)
        except Exception as e:
            print(f"❌ Error initializing Pinecone client/index: {e}")
            return None
    
    return _PINEONE_INDEX

async def retrieve(
    query: str, agent_name: str, k: int = 5
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Asynchronously retrieves the top-k most relevant documents from Pinecone.
    """
    index = _get_pinecone_index(agent_name)
    if index is None:
        return [], None
    
    agent_config = AGENT_CONFIG.get(agent_name.upper())
    namespace = agent_config["namespace"] # Get the agent's namespace

    # 1. Get the query embedding
    query_embedding_list = (await embed_documents([query]))[0]

    # 2. Query Pinecone (blocking call wrapped in to_thread)
    try:
        results = await asyncio.to_thread(
            index.query,
            vector=query_embedding_list,
            top_k=k,
            namespace=namespace, # Use the correct namespace
            include_metadata=True,
            include_values=False
        )
    except Exception as e:
        print(f"❌ Error querying Pinecone index: {e}")
        return [], None

    items = []
    digest_lines = []

    for match in results.get("matches", []):
        score = match.get("score", 0.0)
        metadata = match.get("metadata", {})
        
        # Retrieve the full text chunk from the metadata
        document_text = metadata.get("document_text", "Document text not found in metadata.")

        items.append({
            "id": match.get("id"), 
            "preview": document_text,
            "metadata": metadata, 
            "score": score
        })
        
        digest_lines.append(
            f"- Snippet from {metadata.get('source_file', 'N/A')}: "
            f"{document_text[:240].replace('/n', ' ')} (score={score:.4f})"
        )

    digest = "\n".join(digest_lines) if items else None

    return items, digest