import os
import google.generativeai as genai

# --- API and Model Configuration ---
# GEMINI_API_KEY is read from environment secrets in the deployment platform (e.g., Hugging Face).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# NEW: Pinecone credentials for the external vector database.
PINEONE_API_KEY = os.getenv("PINEONE_API_KEY")
PINEONE_ENVIRONMENT = os.getenv("PINEONE_ENVIRONMENT") # Or PINEONE_HOST for serverless

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "gemini-2.0-flash"

# --- Agent-Specific Configurations ---
# We use a single Pinecone index ("rag-index") and separate data by namespace.
# The dimension must match the EMBED_MODEL (384 for all-MiniLM-L6-v2).
VECTOR_INDEX_NAME = "rag-index" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AGENT_CONFIG = {
    "CEO": {
        "index_name": VECTOR_INDEX_NAME, # All agents use the same index name
        "namespace": "ceo", # Namespace for CEO data
        "data_file": os.path.join(BASE_DIR, "sample_data", "ceo", "conversation.txt"),
    },
    "CTO": {
        "index_name": VECTOR_INDEX_NAME,
        "namespace": "cto", # Namespace for CTO data
        "data_file": os.path.join(BASE_DIR, "sample_data", "cto", "conversation.txt"),
    },
    "CFO": {
        "index_name": VECTOR_INDEX_NAME,
        "namespace": "cfo", # Namespace for CFO data
        "data_file": os.path.join(BASE_DIR, "sample_data", "cfo", "conversation.txt"),
    },
    "CMO": {
        "index_name": VECTOR_INDEX_NAME,
        "namespace": "cmo", # Namespace for CMO data
        "data_file": os.path.join(BASE_DIR, "sample_data", "cmo", "conversation.txt"),
    },
}

# --- Function to Get Generative AI Client ---
def get_genai_client():
    """Initializes and returns the Generative AI client."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please set it as a secret in your deployment environment."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    return genai

# --- End of config.py ---