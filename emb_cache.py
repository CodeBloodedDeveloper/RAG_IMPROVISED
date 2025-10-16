
import shelve
import hashlib
from pathlib import Path

CACHE_PATH = Path("./.emb_cache.db")

def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class EmbeddingCache:
    def __init__(self, path=CACHE_PATH):
        self.path = str(path)

    def __enter__(self):
        # writeback=False for performance; we only set keys explicitly
        self.db = shelve.open(self.path, writeback=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.db.close()
        except Exception:
            pass

    def get(self, text):
        h = doc_hash(text)
        return self.db.get(h)

    def set(self, text, emb):
        h = doc_hash(text)
        self.db[h] = emb

    def bulk_get(self, texts):
        return [self.get(t) for t in texts]

    def bulk_set(self, texts, embeddings):
        for t, e in zip(texts, embeddings):
            self.set(t, e)
