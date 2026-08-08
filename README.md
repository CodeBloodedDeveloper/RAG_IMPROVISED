# Multi-Agent C-Suite RAG Advisor Backend

An asynchronous, high-performance RAG (Retrieval-Augmented Generation) backend powered by **FastAPI**, **Pinecone**, **Sentence-Transformers**, and **Groq (OpenAI SDK)**.

This system orchestrates a panel of AI C-Suite advisors (**Idea Validator**, **CEO**, **CTO**, **CFO**, **CMO**) to guide founders through end-to-end startup validation, technical feasibility, financial unit economics, and go-to-market strategies using isolated vector namespaces and persona-specific system prompts.

---

## 🏗️ System Architecture

```text
               +-------------------------------------------------+
               |              Client Request (/ask)              |
               +-------------------------------------------------+
                                       |
                                       v
               +-------------------------------------------------+
               |             FastAPI / Gunicorn Server           |
               +-------------------------------------------------+
                                       |
                  +--------------------+--------------------+
                  |                                         |
                  v                                         v
   +-----------------------------+           +-----------------------------+
   | Query Rewriter (LLM)        |           | Sentence-Transformer Model  |
   | (Rephrases query w/ context)|           | (all-MiniLM-L6-v2)          |
   +-----------------------------+           +-----------------------------+
                  |                                         |
                  +--------------------+--------------------+
                                       |
                                       v
               +-------------------------------------------------+
               |        Pinecone Vector Search (Namespace)       |
               |       (Retrieves Top-K Relevant Evidence)       |
               +-------------------------------------------------+
                                       |
                                       v
               +-------------------------------------------------+
               |       Groq Llama 3.3 70B Streaming Engine       |
               |    (Persona Prompt + Context + Chat History)    |
               +-------------------------------------------------+
                                       |
                                       v
               +-------------------------------------------------+
               |      FastAPI StreamingResponse (Chunked)        |
               +-------------------------------------------------+
```

---

## ✨ Key Features

* **Multi-Persona C-Suite Guidance:** Dedicated agent personas with domain-isolated system constraints:
  * **Idea Validator:** Comprehensive initial startup vetting.
  * **CEO:** Strategic vision, roadmap, and organizational priorities.
  * **CTO:** System architecture, stack selection, and tech feasibility.
  * **CFO:** Pricing models, unit economics, and burn rate analysis.
  * **CMO:** Go-to-market (GTM) strategy, positioning, and ICP identification.
* **Isolated Vector Retrieval:** Uses Pinecone namespaces (`ceo`, `cto`, `cfo`, `cmo`) under a single index (`rag-index`) to retrieve context specific to the active advisor.
* **Low-Latency Streaming:** Asynchronous chunk streaming via FastAPI's `StreamingResponse` using Groq's high-speed inference endpoints (`llama-3.3-70b-versatile`).
* **Memory-Optimized Deployment:** Configured with Gunicorn process preloading (`--preload`) and lazy model caching to run within memory-restricted environments (e.g., Hugging Face Spaces free tier).
* **Structured JSON Observability:** Custom log formatting outputs single-line JSON logs with execution context, stack traces, and request details.

---

## 🛠️ Tech Stack

| Layer | Technology |
| --- | --- |
| **Framework** | FastAPI, Pydantic v2, Uvicorn, Gunicorn |
| **LLM Engine** | Groq API / OpenAI Python SDK (`llama-3.3-70b-versatile`) |
| **Vector DB** | Pinecone (`pinecone-client` v4+) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384 Dimensions) |
| **Tokenizer / Chunker** | `tiktoken` (`cl100k_base`), Custom overlapping window chunker |
| **Dependency Management** | Poetry, Python 3.11 |

---

## 📂 Repository Structure

```text
├── app.py               # FastAPI application entry point and CORS routing
├── agents.py            # C-Suite personas, context synthesis, and streaming logic
├── retriever.py         # Thread-safe Pinecone connection & vector retrieval
├── embeddings.py        # Sentence-Transformer model loader and document embedder
├── config.py            # Environment configuration, agent namespaces, and LLM client setup
├── chunker.py           # Token-bounded text chunking with overlap
├── ingest.py            # Data ingestion script for embedding generation and upsert
├── logging_config.py    # Custom single-line JSON logging formatter
├── start.sh             # Production startup script running Gunicorn with worker preloading
├── pyproject.toml       # Poetry project dependencies
├── requirements.txt     # Pip dependency fallback
└── sample_data/         # Knowledge base text files per agent domain
```

---

## 🔑 Environment Variables

The application requires the following environment secrets:

| Variable | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | **Yes** | API Key from Groq Console (`console.groq.com`). |
| `PINECONE_API_KEY` | **Yes** | API Key for Pinecone Vector Database. |
| `PINECONE_ENVIRONMENT` | **Yes** | Pinecone environment host/region. |
| `CHAT_MODEL` | No | Defaults to `llama-3.3-70b-versatile`. |
| `LOG_LEVEL` | No | Log verbosity (`INFO`, `DEBUG`, `WARNING`, `ERROR`). |
| `PORT` | No | Binding port for server (Defaults to `7860`). |

---

## 🚀 Local Setup & Installation

### Prerequisites

* Python 3.11+
* Poetry installed (`pip install poetry`)

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/your-username/pyrag.git
cd pyrag
poetry install
```

### 2. Configure Environment Secrets

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_env
LOG_LEVEL=DEBUG
```

### 3. Run Knowledge Base Ingestion (Optional)

If populating Pinecone with initial domain documents:

```bash
poetry run python ingest.py
```

### 4. Start Server

Run via the production shell script:

```bash
chmod +x start.sh
./start.sh
```

Or directly using Uvicorn for local development:

```bash
poetry run uvicorn app:app --reload --port 7860
```

---

## 📡 API Reference

### `GET /`

* **Description:** Health check endpoint to confirm server operational status.
* **Response:**

```json
{
  "message": "✅ Multi-agent AI backend is running"
}
```

### `POST /ask`

* **Description:** Primary query endpoint for chatting with the active C-Suite advisor persona.
* **Payload:**

```json
{
  "activeRole": "CEO",
  "messages": [
    {
      "role": "user",
      "content": "How should I structure my initial fundraising deck?"
    }
  ]
}
```

* **Response:** Raw text chunk stream (`media_type="text/plain"`).
