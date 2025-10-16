# retriever.py

import os
import asyncio
import chromadb
from typing import Optional, Tuple, List, Dict, Any

from config import VECTOR_DB_DIR, AGENT_CONFIG
from embeddings import embed_documents

# --- Global Cache for ChromaDB Collections ---
_CHROMA_COLLECTIONS: Dict[str, Any] = {}

def get_collection(agent_name: str) -> Optional[chromadb.Collection]:
    """
    Initializes and returns a cached ChromaDB collection for a specific agent.
    """
    global _CHROMA_COLLECTIONS
    agent_name_upper = agent_name.upper()

    if agent_name_upper not in _CHROMA_COLLECTIONS:
        if agent_name_upper not in AGENT_CONFIG:
            print(f"⚠️ Agent '{agent_name}' not configured. Retrieval skipped.")
            return None

        collection_name = AGENT_CONFIG[agent_name_upper]["collection_name"]
        agent_db_path = os.path.join(VECTOR_DB_DIR, agent_name_upper.lower())
        os.makedirs(agent_db_path, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=agent_db_path)
        _CHROMA_COLLECTIONS[agent_name_upper] = chroma_client.get_or_create_collection(collection_name)

    return _CHROMA_COLLECTIONS.get(agent_name_upper)

async def retrieve(
    query: str, agent_name: str, k: int = 5
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Asynchronously retrieves the top-k most relevant documents from ChromaDB.
    """
    collection = get_collection(agent_name)
    if collection is None:
        return [], None

    query_embedding = (await embed_documents([query]))[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]

    items = [
        {"id": ids[i], "preview": documents[i], "metadata": metadatas[i], "score": distances[i]}
        for i in range(len(ids))
    ]

    digest_lines = [
        f"- Snippet from {item['metadata'].get('source_file', 'N/A')}: "
        f"{item['preview'][:240].replace('/n', ' ')} (score={item.get('score', 0):.4f})"
        for item in items
    ]
    digest = "\n".join(digest_lines) if items else None

    return items, digest