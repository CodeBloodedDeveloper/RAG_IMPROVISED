import os
import chromadb
from config import VECTOR_DB_DIR, AGENT_CONFIG
from embeddings import embed_documents_local

_CHROMA_COLLECTIONS = {} # Cache for multiple collections

def get_collection(agent_name: str):
    """
    Initializes and returns a ChromaDB collection for a specific agent, caching it globally.
    Returns None if the agent is not configured.
    """
    global _CHROMA_COLLECTIONS
    
    agent_name_upper = agent_name.upper()
    
    if agent_name_upper not in _CHROMA_COLLECTIONS:
        if agent_name_upper not in AGENT_CONFIG:
            print(f"ℹ️ Agent '{agent_name}' not found in AGENT_CONFIG. Retrieval will be skipped.")
            return None
            
        collection_name = AGENT_CONFIG[agent_name_upper]["collection_name"]
        
        agent_db_path = os.path.join(VECTOR_DB_DIR, agent_name_upper.lower())
        os.makedirs(agent_db_path, exist_ok=True)
        
        _chroma_client = chromadb.PersistentClient(path=agent_db_path)
        _CHROMA_COLLECTIONS[agent_name_upper] = _chroma_client.get_or_create_collection(collection_name)
        
    return _CHROMA_COLLECTIONS.get(agent_name_upper)
    
def retrieve(query: str, agent_name: str, k=5, return_digest=True):
    """
    Retrieves top-k relevant chunks from the specified agent's vector store.
    """
    coll = get_collection(agent_name)
    
    if coll is None:
        return [], None
        
    q_emb = embed_documents_local([query])[0]
    results = coll.query(query_embeddings=[q_emb], n_results=k)
    
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]

    items = []
    for i in range(len(ids)):
        items.append({
            "id": ids[i],
            "preview": documents[i],
            "metadata": metadatas[i],
            "score": distances[i]
        })

    if not items:
        return [], None

    digest_lines = []
    for item in items:
        snippet = item['preview'][:240].replace('\n', ' ')
        source_file = item['metadata'].get('source_file', 'N/A')
        score = item.get('score', 0)
        digest_lines.append(f"- Snippet from {source_file}: {snippet} (score={score:.4f})")
    
    digest = "\n".join(digest_lines)
    
    return (items, digest) if return_digest else (items, None)

