# CODE.zip/config.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models & Vector DB config
EMBED_MODEL = "models/embedding-gecko-001"
CHAT_MODEL = "gemini-2.0-flash" # Updated for better performance
VECTOR_DB_DIR = "/tmp/chroma_store" # Base directory for all vector stores

# --- NEW AGENT CONFIGURATION ---
# Central place to define all agents, their data sources, and vector stores.
AGENT_CONFIG = {
    "CMO": {
        "data_file": os.path.join(BASE_DIR, "sample_data", "cmo", "conversations.json"),
        "collection_name": "cmo_agent_data"
    },
    "CFO": {
        "data_file": os.path.join(BASE_DIR, "sample_data", "cfo", "conversations.json"),
        "collection_name": "cfo_agent_data"
    },
    "CTO": {
        "data_file": os.path.join(BASE_DIR, "sample_data", "cto", "conversations.json"),
        "collection_name": "cto_agent_data"
    },
    "CEO": {
        "data_file": os.path.join(BASE_DIR, "sample_data", "ceo", "conversations.json"),
        "collection_name": "ceo_agent_data"
    }
}

def get_genai_client():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai








# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Get the absolute path of the directory where this config.py file is located
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Load environment variables from a .env file
# load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Models & Vector DB config
# EMBED_MODEL = "models/embedding-001"
# CHAT_MODEL = "gemini-2.0-flash"
# # Use an absolute path for the vector database
# VECTOR_DB_DIR = "/tmp/chroma_store"

# def get_genai_client():
#     """
#     Initializes and returns the Generative AI client.
#     """
#     if not GEMINI_API_KEY:
#         raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
#     genai.configure(api_key=GEMINI_API_KEY)
#     return genai