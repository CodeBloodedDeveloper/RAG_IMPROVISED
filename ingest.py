# ingest.py

import json
import uuid
import os
import asyncio
from typing import List, Dict, Any

from embeddings import embed_documents
from chunker import smart_chunk
from retriever import get_collection
from config import AGENT_CONFIG

async def ingest_for_agent(agent_name: str, path: str):
    """
    Asynchronously ingests and processes a JSON file for a specific agent's
    ChromaDB collection.
    """
    collection = get_collection(agent_name)
    if not collection:
        print(f"ℹ️ No collection for agent '{agent_name}'. Skipping.")
        return

    if not os.path.exists(path):
        print(f"❌ Input file not found for {agent_name}: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract text content from the JSON data
    raw_docs = [
        f"Title: {item.get('content', {}).get('title', 'No Title')}\n"
        f"Overview: {item.get('content', {}).get('overview', '')}"
        for item in data if item.get("content")
    ]

    if not raw_docs:
        print(f"⚠️ No documents to process in {path} for '{agent_name}'.")
        return

    # Chunk the documents
    chunked_docs = [chunk for doc in raw_docs for chunk in smart_chunk(doc)]
    if not chunked_docs:
        print(f"⚠️ No chunks generated for '{agent_name}'.")
        return

    # Generate embeddings asynchronously
    vectors = await embed_documents(chunked_docs, batch_size=32)

    # Prepare data for upsertion into ChromaDB
    ids = [str(uuid.uuid4()) for _ in chunked_docs]
    metadatas = [{"source_file": path, "preview": doc[:256]} for doc in chunked_docs]

    # Upsert the data into the collection
    collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=chunked_docs)
    print(f"🚀 Ingested {len(chunked_docs)} chunks for agent '{agent_name}'.")

async def ingest_all_agents_if_needed():
    """
    Iterates through the AGENT_CONFIG and ingests data for each agent if its
    collection is empty. This is an async function.
    """
    ingestion_tasks = []
    for agent_name, config in AGENT_CONFIG.items():
        print(f"\n--- Checking ingestion for agent: {agent_name} ---")
        try:
            collection = get_collection(agent_name)
            # Check if collection exists and is empty
            if collection is not None and collection.count() == 0:
                print(f"Collection for '{agent_name}' is empty. Scheduling ingestion.")
                # Create an async task for each agent's ingestion
                task = ingest_for_agent(agent_name, config["data_file"])
                ingestion_tasks.append(task)
            elif collection is not None:
                print(f"Collection for '{agent_name}' already contains data. Skipping.")
        except Exception as e:
            print(f"❌ Error checking ingestion for {agent_name}: {e}")

    # Run all scheduled ingestion tasks concurrently
    if ingestion_tasks:
        await asyncio.gather(*ingestion_tasks)

