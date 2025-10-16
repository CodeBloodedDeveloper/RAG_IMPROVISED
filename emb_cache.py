# emb_cache.py

import shelve
import hashlib
from pathlib import Path

CACHE_PATH = Path("./.emb_cache.db")

def doc_hash(text: str) -> str:
    """Generates a SHA256 hash for a given text document."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class EmbeddingCache:
    """A simple file-based cache for storing document embeddings."""
    def __init__(self, path=CACHE_PATH):
        self.path = str(path)

    def __enter__(self):
        self.db = shelve.open(self.path, writeback=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.db:
            self.db.close()

    def get(self, text: str):
        return self.db.get(doc_hash(text))

    def set(self, text: str, emb):
        self.db[doc_hash(text)] = emb
