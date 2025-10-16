# Use a lightweight and secure Python base image
FROM python:3.11-slim

# Set environment variables for better Python and Poetry behavior
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1 \
    # Direct the model cache to a known, writable directory
    SENTENCE_TRANSFORMERS_HOME="/app/.cache"

# Install Poetry for dependency management
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set the application's working directory
WORKDIR /app

# Install project dependencies first
COPY pyproject.toml ./
RUN poetry lock
RUN poetry install --no-root

# --- CRITICAL FIX RESTORED ---
# Pre-download and cache the sentence-transformer model during the build process.
# This ensures the model is available offline when the container runs,
# preventing all download-related and permission errors at runtime.
RUN poetry run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# --- END OF CRITICAL FIX ---

# Copy the rest of the application code into the container
COPY . .

# Make the startup script executable
RUN chmod +x ./start.sh

# Expose the port that the Gunicorn server will run on
EXPOSE 7860

# Set the startup script as the final command to run the application
CMD ["./start.sh"]
