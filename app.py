from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from agents import run_agent_conversational
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    A simple lifespan function that runs when the server starts.
    The complex ingestion logic has been moved to a separate script.
    """
    print("--- ✅ FastAPI application is starting up... ---")
    yield
    print("--- 🛑 FastAPI application is shutting down. ---")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class ProviderMessage(BaseModel):
    role: str
    content: str

class AIChatRequest(BaseModel):
    messages: List[ProviderMessage]
    activeRole: str

@app.get("/")
async def root():
    return {"message": "✅ Multi-agent AI backend is running"}

@app.post("/ask")
async def ask_post(request: AIChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages in request.")

    query = request.messages[-1].content
    history = [msg.dict() for msg in request.messages[:-1]]
    role = request.activeRole

    try:
        return await run_agent_conversational(role, query, history)
    except Exception as e:
        print(f"❌ Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error processing the request.")
