---
title: Legal and Policy RAG System
---

# Legal and Policy RAG System

A retrieval-augmented generation system for legal documents, government policy files, enterprise policy materials, and application guidelines.

## Highlights

- Ingested 131 NSFC policy and guideline documents with 587 processed policy chunks.
- Stored and searched 2,590 Milvus vector entities after incremental indexing.
- Combined dense vector retrieval, sparse keyword retrieval, Hybrid RRF fusion, and optional reranking.
- Structured document chunking around policy titles, chapters, clauses, application conditions, materials, and procedures.
- Exposed a FastAPI backend with upload, indexing, retrieval, question answering, and health-check endpoints.
- Added a browser UI with user and administrator workflows for document ingestion and policy Q&A.
- Verified the core project test suite with 59 passing tests and 2 skipped tests.

## Architecture

1. Crawlers collect public policy documents from official sources.
2. Parsers normalize PDF, TXT, Markdown, and DOCX content.
3. Policy-aware chunking preserves chapter, clause, and procedural context.
4. Embedding service generates dense vectors for Milvus indexing.
5. Sparse retrieval builds keyword-oriented evidence recall.
6. Hybrid RRF merges dense and sparse results.
7. Reranking improves final evidence ordering.
8. LLM generation produces grounded answers with source snippets.

## Tech Stack

- Python, FastAPI, Pydantic
- Milvus, Docker Compose
- SiliconFlow BGE embeddings
- Moonshot/Kimi-compatible generation API
- Hybrid RRF retrieval and reranking
- Pytest-based test coverage

## Quick Start

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up -d
pip install -r requirements.txt
uvicorn src.api.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Main Capabilities

- Upload and index legal, policy, and enterprise guideline files.
- Search with dense, sparse, or Hybrid RRF retrieval.
- Return source snippets for traceable answers.
- Track ingestion status, retrieval latency, generated answers, and source count.
- Support policy document crawling and incremental corpus expansion.

## Repository Branches

- `rag-system`: full RAG project source code.
- `main`: intentionally cleared as an empty branch.

