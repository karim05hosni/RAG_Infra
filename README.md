## Index

- [Problem](#problem)
- [What the system does](#what-the-system-does)
    - [Overview (what the system does)](#overview-what-the-system-does)
    - [Ingestion pipeline sequence diagram](#ingestion-pipeline-sequence-diagram)
    - [Multilingual hybrid retrieval sequence diagram](#multilingual-hybrid-retrieval-sequence-diagram)
    - [Faithfulness evaluation & hallucination detection flow](#faithfulness-evaluation--hallucination-detection-flow)
    - [How users interact with it (setup)](#how-users-interact-with-it)
- [Technical challenges](#technical-challenges)
    - [1. Ingestion pipeline optimization under load (batching & I/O)](#1-ingestion-pipeline-optimization-under-load-batching--io)
    - [2. Keyword search evolution (from naive BM25 to PostgreSQL GIN)](#2-keyword-search-evolution-from-naive-bm25-to-postgresql-gin)
    - [3. Multilingual hybrid retrieval & cross-lingual fusion](#3-multilingual-hybrid-retrieval--cross-lingual-fusion)
    - [4. Multimodal video ingestion (speech boundaries vs. text)](#4-multimodal-video-ingestion-speech-boundaries-vs-text)
    - [5. Resilient ground-truth Q&A generation with Write-Ahead Logging (WAL)](#5-resilient-ground-truth-qa-generation-with-write-ahead-logging-wal)
    - [6. Faithfulness evaluation & hallucination detection from scratch](#6-faithfulness-evaluation--hallucination-detection-from-scratch)
    - [7. Conversational AI agent & medical safety boundaries](#7-conversational-ai-agent--medical-safety-boundaries)
- [Architecture decisions](#architecture-decisions)
- [What I’d improve (trade-offs)](#what-id-improve-trade-offs)

---

## Problem

Most production RAG systems suffer from three fundamental issues: naive vector-only search fails on keyword-dense domain jargon (e.g., medical diagnoses or CV skills), cross-lingual content creates retrieval blindspots, and synthetic evaluation pipelines lack failure atomicity and crash resilience. 

I wanted to design an end-to-end hybrid RAG and evaluation infrastructure from first principles—supporting multilingual ophthalmology text and video content, sub-millisecond keyword retrieval, fault-tolerant ground-truth generation, and automated hallucination scoring.

---

## What the system does

### Overview (what the system does)

Knowledge sources (PDFs, DOCX, Video reels) → Transcribed & Chunked → Batched Dense & GIN Indexed → Multilingual Query Expansion & Hybrid Search → Agent Loop with Context Neighbor Expansion → Idempotent Evaluation with WAL & NLI Fact Verification.

#### Ingestion pipeline sequence diagram

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client / API
    participant Fast as FastAPI (Ingestion Service)
    participant FFmpeg as ffmpeg / Whisper
    participant Lang as Language Detector
    participant ST as Embedding Model (bge-m3)
    participant Qdrant as Qdrant Vector DB
    participant PG as PostgreSQL (GIN)

    Client->>Fast: POST /ingest (Files: PDF / DOCX / MP4)
    alt Video File (.mp4)
        Fast->>FFmpeg: Extract audio & transcribe (Whisper)
        FFmpeg-->>Fast: Return timestamped segments
        Fast->>Fast: Chunk by duration (30-second speech windows)
    else Document File (.pdf / .docx / .txt)
        Fast->>Fast: Extract raw text & sentence chunk
    end

    Fast->>Lang: Detect chunk language (en / fr)
    Lang-->>Fast: Return language code + sanitize text
    
    Fast->>Fast: Batch all chunks across multiple files
    Fast->>ST: embed_text(batches, batch_size=N)
    ST-->>Fast: Return dense embeddings

    Fast->>Qdrant: Upsert points (vector + chunk metadata payload)
    Qdrant-->>Fast: Acknowledge vector insertion
    
    Fast->>PG: INSERT INTO sources & chunks (tsvector Simple)
    PG-->>Fast: Acknowledge relational + GIN index commit
    Fast-->>Client: HTTP 200 (Ingestion complete)
```

#### Multilingual hybrid retrieval sequence diagram

```mermaid
sequenceDiagram
    autonumber
    participant User as User / Client
    participant API as FastAPI (Retrieval Service)
    participant Classifier as langid (Classifier)
    participant MT as Helsinki-NLP/opus-mt (en <-> fr)
    participant Qdrant as Qdrant Vector DB
    participant PG as PostgreSQL (GIN tsvector)
    participant RRF as Reciprocal Rank Fusion
    participant Agent as LLM Agent (ReAct)

    User->>API: POST /agent or GET /search (Query)
    API->>Classifier: Detect query language
    Classifier-->>API: Return query_lang ('en' or 'fr')

    par Dense Semantic Search
        API->>Qdrant: Search embedding(query) [Top-K=40]
        Qdrant-->>API: Dense vector candidates & similarity scores
    and Primary Keyword Search
        API->>PG: Full-text search (query, query_lang)
        PG-->>API: Native keyword candidates & ts_rank
    and Cross-Lingual Translation & Keyword Search
        API->>MT: Translate query to target_lang
        MT-->>API: translated_query
        API->>PG: Full-text search (translated_query, target_lang)
        PG-->>API: Translated keyword candidates & ts_rank
    end

    API->>RRF: Fuse ranked lists with weights (w_kw, w_sem, w_trans)
    RRF-->>API: Top fused candidate chunk IDs
    API->>PG: Hydrate chunk payloads & index positions
    PG-->>API: Return chunks text & metadata

    opt Context Neighbor Expansion
        API->>PG: fetch_chunks_in_index_range_by_source(source_id, idx ± 2)
        PG-->>API: Return surrounding chunks for unbroken context
    end

    API->>Agent: Supply contextual chunks into system prompt
    Agent-->>User: Grounded answer + CTA consultation link
```

#### Faithfulness evaluation & hallucination detection flow

```mermaid
flowchart TD
    A["LLM Generated Answer"] --> B["Claim Extraction<br/>(Qwen 2.5 3B via Ollama)"]
    B --> C["Atomic Claims List<br/>[Claim 1, Claim 2, ...]"]
    
    D["Retrieved Context Chunks"] --> E["Atomic Decomposition<br/>(Split multi-fact chunks)"]
    E --> F["Single-Fact Chunks"]

    C & F --> G["NLI Cross-Encoder<br/>(mDeBERTa-v3-base-mnli-xnli)"]
    
    G --> H{"NLI Classification"}
    H -- "Entailment" --> I["Mark Claim as Supported"]
    H -- "Contradiction" --> J["Mark Claim as Contradiction"]
    H -- "Neutral" --> K["LLM-as-a-Judge Fallback<br/>(Qwen 2.5 3B)"]

    K --> L{"Judge Decision"}
    L -- "Entailed" --> I
    L -- "Contradicted / Unsupported" --> J

    I & J --> M["Faithfulness Score<br/>= (Supported Claims) / (Total Claims)"]
```

---

### How users interact with it

#### 1. Prerequisites

- **Docker Desktop** (or Docker Engine + Docker Compose v2).
- **Python 3.13+** with [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `pip`/`venv`.

---

#### 2. Environment configuration

Copy the template environment file:

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Default configuration in `.env`:

```ini
QDRANT_URL='http://localhost:6333'
POSTGRES_URL='postgresql://postgres:postgres@localhost:5432/rag_db'
Ollama_url='http://localhost:11434'
embedding_model='bge-m3'
CORS='http://localhost:5173,http://localhost:3000'
GEMINI_API_KEY='your-optional-key'
```

---

#### 3. Start supporting infrastructure (Docker)

Spin up PostgreSQL, Qdrant, and Ollama in detached mode:

```bash
docker compose up -d
```

What runs automatically:
1. **PostgreSQL** (`rag-postgres`): Creates tables (`sources`, `chunks`, `eval_dataset`, `eval_dataset_wal`, `eval_chunk_status`) and initializes `tsvector` simple dictionaries with GIN indexes via [docker/postgres/init.sql](file:///d:/Projects/RAG_infra/docker/postgres/init.sql).
2. **Qdrant** (`rag-qdrant`): Initializes the vector database engine and web UI on port `6333`.
3. **Ollama** (`rag-ollama`): Boots the local LLM runtime on port `11434`.
4. **Model Puller** (`ollama-pull-model`): Waits for Ollama health check, automatically downloads `bge-m3`, and exits.

Verify container status:
```bash
docker compose ps
```

---

#### 4. Run the FastAPI application

Using `uv` (recommended):
```bash
uv run uvicorn main:app --reload --port 8000
```

Or using an active virtual environment:
```bash
uvicorn main:app --reload --port 8000
```

Interactive Swagger documentation is exposed at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

#### 5. Operational notes

> [!WARNING]
> Running `docker compose down -v` removes persistent named volumes (`rag_postgres_data`, `rag_qdrant_data`, `rag_ollama_data`). Use only when you intend to wipe all indexed vectors and database records.

---

## Technical challenges

### 1. Ingestion pipeline optimization under load (batching & I/O)

When scaling document ingestion to large corpora (tested against 457 documents producing ~8,000 chunks), the initial naive pipeline choked with excessive execution times (~250 seconds) and file-descriptor exhaustion.

#### The bottlenecks:
1. **Per-chunk single encoding**: Calling `SentenceTransformer.encode()` per chunk meant PyTorch spent more time on Python dispatch overhead and memory copying than actual tensor operations. Changing this to batch encode within single files only dropped execution time from ~250s to ~234s.
2. **Inter-file batching**: Individual files often had too few chunks to saturate GPU/CPU SIMD lanes. Accumulating chunks *across files* into large global batches prior to model dispatch utilized hardware parallelization efficiently, slashing runtime down to **~90 seconds** (a **64% reduction**).
3. **OS file handler exhaustion**: Writing each chunk as an isolated JSON file flooded the file system, hitting OS open-file limits. Chunks were consolidated into batched database transactions with byte-size caps (`split_list_by_max_bytes` set to 30 MB chunks) to prevent network socket drops.

| Optimization Stage | Chunks Processed | Total Ingestion Time | Speedup Factor |
| :--- | :--- | :--- | :--- |
| Naive per-chunk `encode()` + JSON files | 8,000 chunks (457 files) | ~250 s | 1.0x (baseline) |
| Intra-file batching | 8,000 chunks (457 files) | ~234 s | 1.07x |
| **Inter-file global batching + single payload I/O** | **8,000 chunks (457 files)** | **~90 s** | **2.78x** |

---

### 2. Keyword search evolution (from naive BM25 to PostgreSQL GIN)

During initial prototyping, custom BM25 was implemented from scratch. While accurate on tiny inputs, it performed repeated runtime scans across all document chunks to compute Term Frequency ($TF$) and Inverse Document Frequency ($IDF$).

#### Complexity comparison:
- **Naive BM25 runtime scan**: In worst-case queries, computational cost scaled at $\mathcal{O}(N^2 \times Q \times T)$ where $N$ is chunk count, $Q$ is query terms, and $T$ is average chunk length. On a modest corpus of 18,000 chunks, query latency was **~7 seconds**.
- **Inverted index data structure**: Computing term frequencies upfront into a dictionary structure `{ term: { chunk_id: frequency } }` removed query-time corpus scanning. Query latency dropped to **~6 milliseconds**.
- **PostgreSQL GIN (`tsvector`)**: To eliminate single-JSON serialization risks (file corruption on crash, heavy JSON rewriting), the inverted index was migrated to PostgreSQL's native Generalized Inverted Index (GIN) using `tsvector` with the `'simple'` dictionary, providing sub-millisecond retrieval with complete ACID transaction guarantees.

---

### 3. Multilingual hybrid retrieval & cross-lingual fusion

The domain involves specialized medical content (ophthalmology clinical articles and surgical reels) published in both English and French. A query written in English must retrieve relevant French clinical notes and vice-versa.

#### The solution:
1. **Evolution of the dense embedding model**: Initial experiments used `paraphrase-multilingual-MiniLM-L12-v2`. However, during evaluation and faithfulness benchmarking, it failed on several specific medical test cases, missing nuanced contextual distinctions. The embedding model was swapped to `bge-m3`, which produced higher retrieval precision and directly improved downstream faithfulness scores.
2. **Machine translation for cross-lingual keyword search**: Dense semantic search alone can miss rare clinical names or exact terminology. During retrieval, the query language is detected using `langid`. The system runs two keyword searches:
   - Search 1: Original query in source language against native chunks.
   - Search 2: Query translated into target language using `Helsinki-NLP/opus-mt-en-fr` or `Helsinki-NLP/opus-mt-fr-en`, querying the opposing language partition.
3. **Weighted Reciprocal Rank Fusion (RRF)**: Results from semantic search, native keyword search, and translated keyword search are combined into a single unified ranking using weighted RRF:

$$\text{RRF Score}(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

Where $k = 60$, $r_m(d)$ is the 1-based rank of document $d$ in system $m$, and $w_m$ allows weighting down translated keyword matches to avoid drift:

```python
# Weighted RRF implementation across 3 search streams
def RRF(keyword_scores, translated_keyword_scores, semantic_scores, k=60, w_kw=1.0, w_sem=1.0, w_translated_kw=1.0, top_k=25):
    fused_scores = {}
    for rank, doc in enumerate(keyword_scores):
        fused_scores[doc['chunk_id']] = fused_scores.get(doc['chunk_id'], 0) + w_kw / (k + rank + 1)
    for rank, doc in enumerate(translated_keyword_scores):
        fused_scores[doc['chunk_id']] = fused_scores.get(doc['chunk_id'], 0) + w_translated_kw / (k + rank + 1)
    for rank, doc in enumerate(semantic_scores):
        fused_scores[doc['chunk_id']] = fused_scores.get(doc['chunk_id'], 0) + w_sem / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

---

### 4. Multimodal video ingestion (speech boundaries vs. text)

Ophthalmology video reels convey critical instructional information. Ingesting video required handling unstructured spoken audio rather than standard written text.

#### The pipeline:
1. **Audio extraction**: `ffmpeg` extracts high-fidelity audio streams from uploaded `.mp4` videos.
2. **Transcription**: OpenAI `Whisper` transcribes audio into timestamped segments.
3. **Speech boundary chunking**: Standard sentence-based or paragraph-based chunkers fail on transcribed speech because continuous spoken discourse lacks grammatical punctuation, producing either run-on chunks or fragments. The chunker shifts to **duration-based sliding windows** (fixed 30-second windows with timestamp metadata `{start_time, end_time}`).
4. **Context preservation**: Chunks are prefixed with their source identity (`From <video_title>: <transcribed_text>`), preserving document lineage for downstream LLM citation.

---

### 5. Resilient ground-truth Q&A generation with Write-Ahead Logging (WAL)

Evaluating RAG systems requires synthetic ground-truth test datasets (question, ground-truth answer, source chunk). Generating these via LLM agents is non-deterministic, slow, and expensive. If the application crashes mid-process (e.g., network drop or power failure), unpersisted LLM calls are wasted.

#### The stateful WAL architecture:
1. **Chunk state tracking**: Chunks transition through explicit statuses in `eval_chunk_status`: `pending` $\to$ `generated` $\to$ `persisted` (with `skipped` or `failed` for edge cases). This provides full visibility into successful and unsuccessful attempts.
2. **Write-Ahead Logging**: When the AI responds, its raw response is immediately saved to `eval_dataset_wal` *before* continuing with parsing and database persistence.
3. **Crash recovery & idempotency**: If the app crashes after the AI responds but before database persistence, the restart procedure reads from the WAL, parses the saved response, and writes it directly to `eval_dataset` without making another AI call.
4. **Automatic WAL cleanup**: Once a chunk's data is safely persisted into `eval_dataset` (or marked skipped), `clean_wal()` automatically deletes that chunk's record from `eval_dataset_wal`. The WAL only holds in-flight or unpersisted crash data, while `eval_chunk_status` keeps the complete audit trail.

```mermaid
stateDiagram-v2
    [*] --> Pending: Chunks registered
    Pending --> Generated: LLM response received
    Generated --> WAL_Stored: Write raw response to eval_dataset_wal
    WAL_Stored --> Persisted: Parse & insert into eval_dataset
    Persisted --> [*]: Automatically cleaned from WAL
    
    WAL_Stored --> Crash: App crashes before DB insert
    Crash --> Recover_From_WAL: App restarts & reads WAL
    Recover_From_WAL --> Persisted: Re-parse & commit without new AI call
```

---

### 6. Faithfulness evaluation & hallucination detection from scratch

To evaluate hallucination rates without external commercial evaluation frameworks, a deterministic faithfulness metric was implemented from scratch:

$$\text{Faithfulness} = \frac{\sum_{i=1}^{C} \mathbb{I}(\text{Claim}_i \text{ is supported})}{\text{Total Extracted Claims } C}$$

#### Implementation details:
1. **Claim extraction**: The LLM answer is decomposed into minimal standalone atomic claims using a locally hosted `Qwen 2.5 3B` model through Ollama, enforcing structured JSON output via Pydantic schemas.
2. **Multi-fact chunk decomposition**: Retrieved chunks often contain multiple facts. Scoring an entire paragraph against a single claim using Natural Language Inference (NLI) frequently yields false "contradictions" or "neutral" scores. Chunks are decomposed into single-fact statements before comparison.
3. **NLI entailment classification**: The zero-shot NLI cross-encoder `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` scores premise-hypothesis pairs into three classes: `entailment`, `neutral`, and `contradiction`.
4. **LLM-as-a-judge for ambiguous neutral pairs**: When NLI outputs `neutral` on complex medical translations, pairs are routed to a local LLM judge (`Qwen 2.5 3B`) with explicit instructions to resolve whether the factual claim is supported by the context.
5. **Connection fault tolerance**: Ollama calls across the local network are wrapped in exponential backoff retry loops to recover from dropped sockets during heavy local inference.

---

### 7. Conversational AI agent & medical safety boundaries

The user-facing agent is built using an autonomous tool-calling loop with multi-turn session memory.

```mermaid
flowchart LR
    User([User Prompt]) --> Memory[Session Memory]
    Memory --> Agent[LLM ReAct Loop]
    Agent --> ToolCall{Needs Retrieval?}
    ToolCall -- Yes --> Tool[hybrid_search]
    Tool --> Expand[get_chunks_in_index_range]
    Expand --> Agent
    ToolCall -- No --> Guardrails[Clinical Boundary Filter]
    Guardrails --> Output([Final User Response])
```

#### Key agent capabilities:
- **Autonomous tool calling**: The agent determines whether to execute `hybrid_search(query)` or request surrounding context via `get_chunks_in_index_range_by_source(source_id, start_index, end_index)` when a retrieved chunk indicates adjacent text is necessary.
- **Session tracking**: Maintains conversational state per `sessionId` with bounded history windows to prevent token context bloat.
- **Medical safety & prompt injection defense**:
  - All incoming user input is wrapped inside `<userInput>` tags and explicitly marked untrusted in the system instructions.
  - The agent is strictly barred from offering medical diagnoses or personalized assessments: personal symptom descriptions immediately trigger an empathetic refusal and direct the user to book an appointment with Dr. Amr.

---

## Architecture decisions

| Component | Selected Technology | Alternative Considered | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **API Framework** | **FastAPI (Python 3.13)** | Flask / Django | Asynchronous request handling, native Pydantic schema validation, and OpenAPI documentation. |
| **Relational & Search** | **PostgreSQL (`tsvector` + GIN)** | Elasticsearch / SQLite | Eliminates maintaining a separate search cluster. PostgreSQL GIN provides sub-millisecond keyword search and transactional consistency. |
| **Vector Engine** | **Qdrant** | Pinecone / FAISS | In-memory HNSW vector search, native metadata filtering, high-throughput batch upserts, and clean local Docker footprint. |
| **Local Embedding Model** | **`bge-m3`** | `paraphrase-multilingual-MiniLM-L12-v2` | Swapped after MiniLM underperformed on faithfulness evaluation test cases; `bge-m3` provides stronger multilingual representation. |
| **Local LLM Runtime** | **Ollama** | Cloud APIs exclusively | Enables local offline execution of embedding models (`bge-m3`) and small reasoning models (`Qwen 2.5 3B`). |
| **Cross-Lingual MT** | **`Helsinki-NLP/opus-mt`** | LLM-based translation | Dedicated translation models (`opus-mt-en-fr` and `opus-mt-fr-en`) are faster, deterministic, and avoid hallucinated terminology on medical phrases. |
| **NLI Classifier** | **`mDeBERTa-v3-base-mnli-xnli`** | Embedding similarity | Embedding cosine distance measures semantic relatedness rather than factual entailment. NLI accurately checks if a premise supports or contradicts a claim. |

---

## What I’d improve (trade-offs)

- **Synchronous query translation latency**: Translating user queries via `Helsinki-NLP/opus-mt` occurs synchronously during the retrieval request. Caching frequent query translations would avoid repeatedly translating identical queries across sessions.
- **Asynchronous background task workers**: Video audio extraction (`ffmpeg`) and transcription (`Whisper`) run directly within the application request flow. Moving heavy media ingestion to an asynchronous worker queue would prevent blocking the web server during long uploads.
- **Cross-encoder re-ranking**: Adding a second-stage cross-encoder re-ranker after RRF fusion would further refine top-3 chunk ordering before passing context to the LLM agent.
