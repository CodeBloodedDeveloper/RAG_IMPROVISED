# embeddings.py

import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

# --- Global Cache for the Embedding Model ---
_EMBEDDING_MODEL = None

def get_embedding_model():
    """
    Loads and caches the SentenceTransformer model using a lazy loading pattern.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            _EMBEDDING_MODEL = SentenceTransformer(EMBED_MODEL)
        except ImportError:
            raise RuntimeError("sentence-transformers is not installed.")
    return _EMBEDDING_MODEL

async def embed_documents(docs: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Asynchronously generates embeddings for a list of documents.
    """
    model = get_embedding_model()
    # Run the CPU-bound encoding task in a separate thread to avoid blocking the event loop
    embeddings = await asyncio.to_thread(
        model.encode, docs, show_progress_bar=False, convert_to_numpy=True
    )
    return [e.tolist() for e in embeddings]