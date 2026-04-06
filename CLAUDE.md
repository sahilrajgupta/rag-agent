# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Create a `.env` file with:
```
ANTHROPIC_API_KEY=your_key_here
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn api:app --reload
```

## Ingesting a PDF directly

```bash
python ingest.py path/to/document.pdf
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) agent with three components:

- **[ingest.py](ingest.py)** — Loads a PDF, splits it into 500-char chunks (50 overlap), embeds with `all-MiniLM-L6-v2` (HuggingFace, runs locally), and persists to ChromaDB at `chroma_db/`.
- **[agent.py](agent.py)** — Loads the ChromaDB vector store, retrieves top-4 similar chunks, and runs a `RetrievalQA` chain using `claude-sonnet-4-20250514` via `langchain-anthropic`. Returns answer + source page citations.
- **[api.py](api.py)** — FastAPI wrapper exposing `POST /upload` (ingest a PDF), `POST /ask` (question answering), and `GET /health`.

**Data flow:** PDF upload → `ingest.py` chunks & embeds → ChromaDB → `agent.py` retrieves chunks → Claude LLM answers with citations → API response.

The vector store persists at `chroma_db/` (local directory). Uploaded PDFs are saved to `uploads/` before ingestion. Both directories are created at runtime.
