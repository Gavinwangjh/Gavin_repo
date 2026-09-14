# Legal & Policy RAG System

A retrieval-augmented generation system for legal documents, government policies, enterprise policy documents, and grant/application guidelines. The project integrates document crawling, legal/policy-aware chunking, vector indexing, hybrid retrieval, reranking, source citation, API services, and a web interface.

## Highlights

- Built for legal and policy documents, with structure-aware chunking for articles, sections, application conditions, required materials, process steps, funding rules, and notices.
- Supports document ingestion for PDF, TXT, Markdown, and DOCX.
- Uses Milvus as the vector database with 1024-dimensional embeddings.
- Combines dense vector retrieval, sparse TF-IDF retrieval, Hybrid RRF fusion, and policy-aware reranking.
- Provides FastAPI endpoints for upload, query, health checks, statistics, and knowledge-base management.
- Includes an admin/user oriented web interface for document indexing and policy Q&A.
- Supports incremental indexing with document hash detection to avoid duplicate ingestion.

## Current Metrics

The following metrics were collected from the local running system:

| Area | Metric |
| --- | ---: |
| Crawled NSFC policy pages | 6 pages |
| Crawled policy documents | 131 documents |
| Preprocessed policy chunks | 587 chunks |
| Average chunks per policy document | 4.48 |
| API-reported documents | 159 documents |
| Milvus vector entities | 2590 chunks |
| Embedding dimension | 1024 |
| Milvus collection | `document_collection` |
| Vector index | `IVF_FLAT` |
| Similarity metric | `COSINE` |
| Sparse index coverage | 1466 chunks |
| TF-IDF features | 500 |
| Incremental rerun result | 131 success/skipped, 0 failed |
| Hybrid RRF candidates | 50 dense + 50 sparse |
| Final retrieved sources | Top-5 |
| API retrieval time in sample query | about 4.12s |
| API total time in sample query | about 4.96s |
| Milvus vector search time | about 0.02s |
| Top-5 rerank time | about 0.001s-0.003s |
| Regression tests | 59 passed, 2 skipped |

Note: LLM generation requires a valid Moonshot/Kimi API key. Retrieval and source citation are functional independently; generation will fail with `401 Invalid Authentication` if the API key is invalid.

## Architecture

```text
Document sources
  -> crawler / upload
  -> document loader
  -> legal & policy chunking
  -> embedding
  -> Milvus vector store
  -> dense retrieval + sparse retrieval
  -> Hybrid RRF fusion
  -> policy reranker
  -> RAG prompt
  -> LLM answer with source chunks
```

## Tech Stack

- Backend: Python, FastAPI, Click
- Web UI: Streamlit, HTML/CSS/JavaScript static app
- Vector database: Milvus, etcd, MinIO, Attu
- Embedding: SiliconFlow `BAAI/bge-large-zh-v1.5`
- LLM: Moonshot/Kimi compatible API
- Retrieval: Dense vector search, TF-IDF sparse search, Hybrid RRF
- Reranking: Policy-document rule reranker, score reranker
- Crawling: BeautifulSoup, aiohttp, lxml
- Testing: Pytest
- Deployment: Docker Compose

## Project Structure

```text
.
├── config/                 # Project configuration
├── docker/                 # Docker Compose services for Milvus stack
├── examples/               # Usage examples
├── scripts/                # Utility scripts, including policy crawling
├── src/
│   ├── api/                # FastAPI application
│   ├── cli/                # Command-line tools
│   ├── document_loader/    # Document parsing and chunking
│   ├── embedding/          # Embedding clients and cache
│   ├── evaluation/         # Evaluation utilities
│   ├── generation/         # LLM client and RAG generation
│   ├── knowledge_base/     # Incremental sync and KB state management
│   ├── rerank/             # Reranking strategies
│   ├── retrieval/          # Dense/sparse/hybrid retrieval
│   ├── scrapers/           # Policy crawlers
│   ├── utils/              # Config and helper utilities
│   ├── vector_store/       # Milvus integration
│   └── web/                # Web interface
├── tests/                  # Unit and integration tests
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick Start

### 1. Create environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```env
MOONSHOT_API_KEY=your-moonshot-api-key
SILICONFLOW_API_KEY=your-siliconflow-api-key
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 3. Start Milvus

```bash
docker compose -f docker/docker-compose.yml up -d
```

Milvus Attu UI:

```text
http://localhost:3000
```

### 4. Start API service

```bash
python -X utf8 -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### 5. Start Streamlit web UI

```bash
python -m streamlit run src/web/streamlit_app.py
```

## Common Commands

### Upload documents

```bash
python -X utf8 -m src.cli.cli upload data/sample_docs --mode incremental
```

### Query the RAG system

```bash
python -X utf8 -m src.cli.cli query "What are the application conditions?" --top-k 5 --method hybrid_rrf
```

### List knowledge-base stats

```bash
python -X utf8 -m src.cli.cli list-docs
```

### Crawl NSFC policy documents

```bash
python -X utf8 scripts/scrape_nsfc_policies.py --pages 6 --delay 0.8
```

### Upload crawled policy documents

```bash
python -X utf8 -m src.cli.cli upload data/nsfc_policy_docs_2025 --mode incremental
```

## API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | GET | Service health check |
| `/stats` | GET | System, retriever, generator stats |
| `/knowledge/stats` | GET | Knowledge-base statistics |
| `/documents/upload` | POST | Upload and index documents |
| `/query` | POST | RAG query |

Example query:

```bash
curl -X POST "http://127.0.0.1:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What are the application conditions?\",\"top_k\":5,\"search_method\":\"hybrid_rrf\"}"
```

## Retrieval Design

The retrieval pipeline contains four stages:

1. Dense vector search in Milvus.
2. Sparse TF-IDF retrieval for keyword and article/section matching.
3. Hybrid RRF fusion to merge dense and sparse results.
4. Policy-aware reranking based on legal/policy structure and domain terms.

This design is suitable for legal and policy documents where exact wording, section titles, application conditions, and process requirements matter.

## Testing

```bash
python -X utf8 -m pytest tests/test_generation.py tests/test_api.py tests/test_retrieval.py -q
```

Latest local result:

```text
59 passed, 2 skipped
```

## Notes

- Do not commit `.env`, local vector data, crawled corpora, or virtual environments.
- The repository includes `.env.example` for configuration.
- The current default LLM endpoint is `https://api.moonshot.ai/v1` with model `kimi-k3`.
- If the LLM API key is invalid, retrieval still works, but answer generation returns an authentication error.
