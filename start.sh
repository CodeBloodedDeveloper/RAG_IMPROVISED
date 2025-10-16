#!/bin/bash
# This script prepares the environment and starts the production server.

# Exit immediately if any command fails, preventing a partial or broken startup.
set -e

# --- Step 1: Run One-Time Data Ingestion ---
# This script runs ONCE to build the vector database if it doesn't exist.
# This happens BEFORE the web server workers are started, preventing race conditions.
echo "--- 🚀 Running one-time data ingestion check... ---"
poetry run python run_ingestion.py

# --- Step 2: Start the Production Server ---
# Now that the database is guaranteed to exist, we can safely start Gunicorn.
# Each worker will start quickly because they don't need to run ingestion.
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