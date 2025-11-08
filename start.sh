#!/bin/bash
# This script prepares the environment and starts the production server.

# Exit immediately if any command fails.
set -e

# --- Step 1: No more local DB check or ingestion needed ---
echo "--- ✅ External Vector DB configured. ---" # NEW/MODIFIED STATUS MESSAGE

# --- Step 2: Start the Production Server ---
# Workers will now connect to Pinecone on first request.
echo "--- ✅ Setup complete. Starting Gunicorn server... ---"

# Use Gunicorn to run the FastAPI app with multiple workers for scalability.
# --max-requests 500: Automatically restarts a worker after it handles 500 requests.
exec poetry run gunicorn -w 2 -k uvicorn.workers.UvicornWorker \
  --max-requests 500 \
  --bind "0.0.0.0:${PORT:-7860}" \
  --log-level info \
  --access-logfile - \
  --error-logfile - \
  app:app