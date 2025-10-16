import os

# --- API and Model Configuration ---
# In a production environment, environment variables are set directly by the host.
# There is no need to load a .env file.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "gemini-2.0-flash"

# --- Vector Database Configuration ---
VECTOR_DB_DIR = "/app/chroma_store"


# --- Agent-Specific Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_CONFIG = {
    "CEO": {
        "collection_name": "ceo_agent_data",
        "data_file": os.path.join(BASE_DIR, "sample_data", "ceo", "conversation.txt"),
    },
    "CTO": {
        "collection_name": "cto_agent_data",
        "data_file": os.path.join(BASE_DIR, "sample_data", "cto", "conversation.txt"),
    },
    "CFO": {
        "collection_name": "cfo_agent_data",
        "data_file": os.path.join(BASE_DIR, "sample_data", "cfo", "conversation.txt"),
    },
    "CMO": {
        "collection_name": "cmo_agent_data",
        "data_file": os.path.join(BASE_DIR, "sample_data", "cmo", "conversation.txt"),
    },
}

# --- Function to Get Generative AI Client ---
def get_genai_client():
    """Initializes and returns the Generative AI client."""
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please set it as a secret in your deployment environment."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    return genai