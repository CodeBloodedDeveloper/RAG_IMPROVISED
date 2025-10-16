import json
import uuid
import os  # <-- THE MISSING IMPORT
from embeddings import embed_documents_with_cache
from chunker import smart_chunk
from retriever import get_collection
from config import AGENT_CONFIG

def ingest_for_agent(agent_name: str, path: str):
    """
    Ingests a JSON file into a specific agent's ChromaDB collection.
    """
    collection = get_collection(agent_name)
    if not collection:
        print(f"ℹ️ No collection configured for agent '{agent_name}'. Skipping ingestion.")
        return

    if not os.path.exists(path):
        print(f"❌ Input file not found for agent {agent_name}: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_docs = []
    for item in data:
        content = item.get("content")
        if not content or not content.get("title"):
            continue

        title = content.get("title", "No Title")
        overview = content.get("overview", "")
        takeaway = content.get("takeaway", "")
        guest = content.get("podcast_details", {}).get("guest", "N/A")
        doc_text = f"Title: {title}\nGuest: {guest}\nOverview: {overview}\nTakeaway: {takeaway}"

        insights_text = ""
        insights_key = "key_insights" if "key_insights" in content else "key_themes"
        for insight in content.get(insights_key, []):
            heading = insight.get("heading", "")
            points = "\n- ".join(insight.get("points", []))
            if heading and points:
                insights_text += f"\n\nInsight: {heading}\n- {points}"

        if insights_text:
            doc_text += insights_text
        raw_docs.append(doc_text)
    
    if not raw_docs:
        print(f"⚠️ No documents to process in {path} for agent '{agent_name}'.")
        return

    chunked_docs, chunk_sources = [], []
    for i, d in enumerate(raw_docs):
        original_item = next((item for item in data if item.get("content", {}).get("title") in d), None)
        chunks = smart_chunk(d, max_tokens=500, overlap_tokens=100)
        for c_idx, c in enumerate(chunks):
            chunked_docs.append(c)
            source_file = original_item.get("file_name", path) if original_item else path
            chunk_sources.append({"source_file": source_file, "index": i, "chunk_index": c_idx})

    if not chunked_docs:
        print(f"⚠️ No chunks generated for agent '{agent_name}'.")
        return

    ids = [str(uuid.uuid4()) for _ in chunked_docs]
    vectors = embed_documents_with_cache(chunked_docs, use_local=True, batch_size=32)
    previews = [d[:512] for d in chunked_docs]
    metadatas = [{**src, "preview": prev} for src, prev in zip(chunk_sources, previews)]

    collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=previews)
    print(f"🚀 Ingested {len(previews)} chunk(s) for agent '{agent_name}'.")

def ingest_all_agents_if_needed():
    """
    Iterates through the AGENT_CONFIG and ingests data for each agent if its collection is empty.
    """
    for agent_name, config in AGENT_CONFIG.items():
        print(f"\n--- Checking ingestion for agent: {agent_name} ---")
        try:
            collection = get_collection(agent_name)
            if collection is not None and collection.count() == 0:
                print(f"Collection for '{agent_name}' is empty. Starting ingestion.")
                ingest_for_agent(agent_name, config["data_file"])
            elif collection is not None:
                print(f"Collection for '{agent_name}' already contains data. Skipping.")
        except Exception as e:
            print(f"❌ Error during ingestion for {agent_name}: {e}")

