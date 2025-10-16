import numpy as np
from config import get_genai_client

_LOCAL_MODEL = None

def get_local_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads and caches the SentenceTransformer model.
    """
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("sentence-transformers is not installed. Please add it to requirements.txt")
        _LOCAL_MODEL = SentenceTransformer(model_name)
    return _LOCAL_MODEL

def embed_documents_local(docs, batch_size=32):
    """
    Return list of embeddings (as lists) for provided docs using local model.
    """
    model = get_local_model()
    # model.encode is a method of the SentenceTransformer object
    emb = model.encode(docs, show_progress_bar=False, convert_to_numpy=True)
    # convert to native Python lists for serialization compatibility
    return [e.tolist() for e in emb]

def embed_documents_with_cache(docs, use_local=True, batch_size=32):
    """
    Embeds a list of documents using either a local or a remote model.
    """
    if use_local:
        return embed_documents_local(docs, batch_size=batch_size)
    else:
        return embed_documents_remote(docs)


