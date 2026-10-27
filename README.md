# RAG Infrastructure Backend

This repository contains the backend infrastructure for a hybrid Retrieval-Augmented Generation (RAG) system with support for full-text search, dense vector retrieval, cross-lingual querying, and LLM evaluations.

---

## Architecture Overview

The system consists of the following components:

| Component | Role | Host URL / Port | Container Name |
| :--- | :--- | :--- | :--- |
| **FastAPI** | Main Application Backend (runs on host) | `http://localhost:8000` | *(Not containerized)* |
| **Ollama** | Local Embedding & LLM Server (serves `bge-m3`) | `http://localhost:11434` | `rag-ollama` |
| **Qdrant** | Vector Database (dense embeddings search + dashboard) | `http://localhost:6333` | `rag-qdrant` |
| **PostgreSQL** | Relational DB + Full-Text Search (`tsvector`) | `localhost:5432` | `rag-postgres` |

Persistent data for Ollama models, Qdrant vectors, and PostgreSQL tables are preserved using Docker named volumes.

---

## Setup Plan & Quickstart Guide

This setup is designed to be minimal, clean, and easily maintainable without requiring deep DevOps knowledge.

### Prerequisites

1. **Docker Desktop** (or Docker Engine + Docker Compose v2). Ensure Docker is running.
2. **Python 3.13+** with [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `pip`/`venv`.

---

### Step 1: Configure Environment Variables

Create your local `.env` file from the provided template:

```bash
# On Windows (PowerShell)
Copy-Item .env.example .env

# On Linux / macOS
cp .env.example .env
```

Review `.env` to verify default connection settings:
```ini
QDRANT_URL='http://localhost:6333'
POSTGRES_URL='postgresql://postgres:postgres@localhost:5432/rag_db'
Ollama_url='http://localhost:11434'
embedding_model='bge-m3'
CORS='http://localhost:5173,http://localhost:3000'
```
*(Add your `GEMINI_API_KEY` or `OPENAI_API_KEY` if you plan to use hosted LLM features).*

---

### Step 2: Start the Supporting Services (Docker)

Start PostgreSQL, Qdrant, and Ollama in detached mode:

```bash
docker compose up -d
```

#### What happens automatically:
1. **PostgreSQL** starts and initializes the database schema (`sources`, `chunks`, `eval_*`) from [docker/postgres/init.sql](file:///d:/Projects/RAG_infra/docker/postgres/init.sql).
2. **Qdrant** starts and exposes the vector engine and web dashboard.
3. **Ollama** starts the server on port `11434`.
4. **`ollama-pull-model`** helper container waits for Ollama to become healthy, automatically downloads the `bge-m3` embedding model, and exits cleanly.

---

### Step 3: Verify Container Health & Downloaded Models

1. Check that all containers are healthy:
   ```bash
   docker compose ps
   ```

2. Confirm that the `bge-m3` model is installed in Ollama:
   ```bash
   docker exec -it rag-ollama ollama list
   ```
   You should see `bge-m3:latest` listed in the output.

3. *(Optional)* Open the Qdrant Web UI in your browser:
   [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

### Step 4: Run the FastAPI App Locally

Run the application on your host machine without containerization:

Using `uv` (recommended):
```bash
uv run uvicorn main:app --reload --port 8000
```

Or using an active virtual environment:
```bash
# If using standard venv:
# .venv\Scripts\activate   (Windows)
# source .venv/bin/activate (Linux/Mac)
uvicorn main:app --reload --port 8000
```

Once running, access the interactive API docs at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Test Ollama connection endpoint: [http://localhost:8000/test-ollama-connection](http://localhost:8000/test-ollama-connection)

---

## Daily Operations & Useful Commands

### Stopping the Services
```bash
docker compose stop
```
*(This pauses the containers without deleting any database records or downloaded models).*

### Restarting the Services
```bash
docker compose start
```

### Viewing Service Logs
```bash
# View all logs
docker compose logs -f

# View logs for a specific service (e.g. ollama or postgres)
docker compose logs -f ollama
docker compose logs -f postgres
```

### Manually Pulling or Updating Models in Ollama
If you ever want to pull additional models or update `bge-m3` manually:
```bash
docker exec -it rag-ollama ollama pull bge-m3
```

### Resetting Everything (Clean State)
If you want to completely wipe database data and re-download models from scratch:
```bash
docker compose down -v
```
> [!WARNING]
> The `-v` flag removes the Docker volumes (`rag_postgres_data`, `rag_qdrant_data`, `rag_ollama_data`). Use only when you intend to reset to a completely clean slate.

---

## Directory Structure

```text
├── app/                      # Application source code
│   ├── agent/                # RAG agents & workflows
│   ├── clients/              # Postgres & Ollama connection clients
│   ├── controllers/          # API route definitions
│   ├── eval/                 # Evaluation dataset & quality pipelines
│   ├── ingestion/            # Readers, chunkers, and embedding ingestion
│   ├── repositories/         # Postgres & Qdrant database access layers
│   └── retrieval/            # BM25, semantic, and hybrid search services
├── docker/
│   └── postgres/
│       └── init.sql          # Auto-executed on first database container startup
├── .env.example              # Environment variables template
├── docker-compose.yml        # Multi-container setup for Ollama, Qdrant, Postgres
├── main.py                   # FastAPI entrypoint (runs on host)
├── pyproject.toml            # Python dependencies (uv / pip)
└── README.md                 # Setup plan & documentation
```
