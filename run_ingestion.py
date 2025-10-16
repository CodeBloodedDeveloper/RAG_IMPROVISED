import asyncio
import os
from config import VECTOR_DB_DIR
from ingest import ingest_all_agents_if_needed

async def main():
    """
    Entry point for the one-time data ingestion process.
    Checks if the database exists and builds it if it doesn't.
    """
    print("--- 🚀 Starting One-Time Data Ingestion Script ---")
    if not os.path.exists(VECTOR_DB_DIR):
        print(f"Database not found at {VECTOR_DB_DIR}. Starting ingestion...")
        await ingest_all_agents_if_needed()
        print("--- ✅ Data Ingestion Complete ---")
    else:
        print("--- ✅ Database already exists. Skipping ingestion. ---")

if __name__ == "__main__":
    asyncio.run(main())
