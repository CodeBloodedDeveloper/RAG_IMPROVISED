# maintenance.py

import os
import shutil
from config import VECTOR_DB_DIR, EMBED_MODEL
from emb_cache import CACHE_PATH

def clean_vector_stores():
    """Deletes the ChromaDB storage directory."""
    if os.path.exists(VECTOR_DB_DIR):
        try:
            shutil.rmtree(VECTOR_DB_DIR)
            print("✅ Removed vector store.")
        except OSError as e:
            print(f"❌ Error removing vector store: {e.strerror}")

def clean_embedding_cache():
    """Deletes the shelve embedding cache files."""
    for ext in ['', '.bak', '.dat', '.dir']:
        filepath = str(CACHE_PATH) + ext
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"❌ Error removing cache file {filepath}: {e.strerror}")
    print("✅ Cleaned embedding cache.")

def verify_or_download_embedding_model():
    """Verifies and downloads the embedding model if needed."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Verifying model: '{EMBED_MODEL}'...")
        SentenceTransformer(EMBED_MODEL)
        print("✅ Embedding model is available.")
    except Exception as e:
        print(f"❌ Could not load embedding model: {e}")

if __name__ == "__main__":
    print("\n--- 🚀 Running Maintenance Script ---")
    clean_vector_stores()
    clean_embedding_cache()
    verify_or_download_embedding_model()
    print("\n--- ✅ Maintenance Complete ---")
