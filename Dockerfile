# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Set a writeable cache directory for sentence-transformers models
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/.cache

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install the packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- CRITICAL STEP: Download and cache the model during the build ---
# This runs only once when the image is created, ensuring the model is available offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of your application code into the container
COPY . /code/

# Tell the port that your app will run on
EXPOSE 7860

# --- FINAL COMMAND ---
# Run the Gunicorn server directly. The application's own startup logic
# will handle the one-time data ingestion.
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860", "--log-level", "debug", "--access-logfile", "-", "--error-logfile", "-", "app:app"]

