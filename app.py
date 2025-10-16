# CODE.zip/app.py

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from agents import run_agent_conversational
from fastapi.middleware.cors import CORSMiddleware
from ingest import ingest_all_agents_if_needed
from config import AGENT_CONFIG, VECTOR_DB_DIR

# --- LIFESPAN FUNCTION TO RUN ON STARTUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once when the application starts.
    It checks if the vector databases exist and ingests data for all agents if needed.
    """
    print("--- Application starting up... ---")
    try:
        # Check if the parent chroma_store directory exists
        if not os.path.exists(VECTOR_DB_DIR) or not any(AGENT_CONFIG.keys()):
            print(f"Database not found or agent config empty at {VECTOR_DB_DIR}.")
            print("--- Starting one-time data ingestion for all agents... ---")
            ingest_all_agents_if_needed()
            print("--- Ingestion complete. Application is ready. ---")
        else:
            print("--- Database found for agents. Skipping ingestion. ---")
    except Exception as e:
        print(f"FATAL: An error occurred during startup ingestion: {e}", file=sys.stderr)
        sys.exit(1)
        
    yield
    print("--- Application shutting down. ---")

# Initialize the FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

# --- Add CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEW Pydantic Models to match Frontend ---
class ProviderMessage(BaseModel):
    role: str
    content: str
    roleContext: Optional[str] = None

class AIChatRequest(BaseModel):
    messages: List[ProviderMessage]
    activeRole: str
    sessionId: Optional[str] = None
    userId: Optional[str] = None

# --- Root Endpoint ---
@app.get("/")
def root():
    return {"message": "✅ Multi-agent AI backend running"}

# --- MODIFIED Main Endpoint ---
@app.post("/ask")
def ask_post(request: AIChatRequest):
    """
    Handles chat requests from the frontend, now with conversation history.
    """
    # Separate the last message (the current query) from the rest of the history
    if not request.messages:
        return {"error": "No messages found in the request."}

    query = request.messages[-1].content
    history = [msg.dict() for msg in request.messages[:-1]] # Pass all but the last message as history

    role = request.activeRole
    
    # Call the new conversational function
    result = run_agent_conversational(role, query, history)
    return result