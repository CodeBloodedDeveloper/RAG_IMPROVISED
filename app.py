# app.py

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from agents import run_agent_conversational
from ingest import ingest_all_agents_if_needed
from config import VECTOR_DB_DIR
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager to handle application startup and shutdown events.
    On startup, it triggers the data ingestion process.
    """
    print("--- Application starting up... ---")
    try:
        # Check if the database directory exists. If not, ingest data.
        if not os.path.exists(VECTOR_DB_DIR):
            print(f"Database not found at {VECTOR_DB_DIR}. Starting one-time ingestion...")
            await ingest_all_agents_if_needed()
            print("--- Ingestion complete. Application is ready. ---")
        else:
            print("--- Database found. Skipping ingestion. ---")
    except Exception as e:
        print(f"FATAL: An error occurred during startup ingestion: {e}", file=sys.stderr)
        # Exit if ingestion fails, as the app cannot function correctly.
        sys.exit(1)

    yield
    print("--- Application shutting down. ---")

# Initialize the FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

# Add CORS Middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models for Request Validation ---
class ProviderMessage(BaseModel):
    role: str
    content: str

class AIChatRequest(BaseModel):
    messages: List[ProviderMessage]
    activeRole: str

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint to check if the service is running."""
    return {"message": "✅ Multi-agent AI backend is running"}

@app.post("/ask")
async def ask_post(request: AIChatRequest):
    """
    Handles chat requests asynchronously, processes them with the conversational
    agent, and returns the AI's response.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages found in the request.")

    # Extract the current query and the conversation history
    query = request.messages[-1].content
    history = [msg.dict() for msg in request.messages[:-1]]
    role = request.activeRole

    # Await the agent's response without blocking the server
    try:
        result = await run_agent_conversational(role, query, history)
        return result
    except Exception as e:
        print(f"❌ Error during agent execution: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

