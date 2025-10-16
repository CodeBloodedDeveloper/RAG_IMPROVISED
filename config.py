import os
from dotenv import load_dotenv

# Load environment variables from a .env file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# --- API and Model Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # More efficient model
CHAT_MODEL = "gemini-2.0-flash"  # Use a more recent, efficient model

# --- Vector Database Configuration ---
VECTOR_DB_DIR = "/tmp/chroma_store"
AGENT_CONFIG = {
    "CEO": {
        "collection_name": "ceo_agent_data",
        "data_file": os.path.join(os.path.dirname(__file__), "sample_data", "ceo", "conversations.json"),
    },
    "CTO": {
        "collection_name": "cto_agent_data",
        "data_file": os.path.join(os.path.dirname(__file__), "sample_data", "cto", "conversations.json"),
    },
    "CFO": {
        "collection_name": "cfo_agent_data",
        "data_file": os.path.join(os.path.dirname(__file__), "sample_data", "cfo", "conversations.json"),
    },
    "CMO": {
        "collection_name": "cmo_agent_data",
        "data_file": os.path.join(os.path.dirname(__file__), "sample_data", "cmo", "conversations.json"),
    },
}

# --- Function to Get Generative AI Client ---
def get_genai_client():
    """Initializes and returns the Generative AI client."""
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai