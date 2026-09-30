# RAG Document Assistant

A conversational question-answering assistant over your own documents. It combines
semantic and keyword search, answers **only** from the retrieved text with citations,
refuses when the documents don't contain the answer, and remembers the conversation.

**Stack:** Python · LangChain · LangGraph · Hugging Face (sentence-transformers, cross-encoder) ·
ChromaDB · BM25 · FastAPI · Streamlit · Ollama / OpenAI · pytest · Docker

## Architecture

```
                      ┌──────────── INGESTION ────────────┐
PDF / MD / TXT ─► load (page metadata) ─► chunk (800 chars, 120 overlap) ─► embed (MiniLM) ─► ChromaDB
                                                                                   │
                                                                   BM25 keyword index (in memory)

                      ┌──────────── CHAT (LangGraph agent) ────────────┐
Streamlit UI ─► FastAPI /chat ─► rewrite follow-up ─► route ─┬─► chitchat reply
                                                            └─► hybrid retrieve ─┬─► generate answer + [n] citations
                                                                   ▲            ├─► reformulate query ─┐
                                                                   └────────────┼──────────────────────┘
                                                                                └─► refuse ("not in documents")
hybrid retrieve = vector search + BM25 ─► Reciprocal Rank Fusion ─► (optional) cross-encoder re-rank
```

## Features by phase
| Phase | What | Where |
|---|---|---|
| 1 | Ingestion, idempotent chunk IDs, grounded prompt with citations, REST API, tests | `app/ingest.py`, `app/rag.py`, `app/main.py` |
| 2 | LangGraph agent: follow-up rewriting, routing, retry with reformulation, per-session memory, Streamlit chat UI | `app/agent.py`, `ui/streamlit_app.py` |
| 3 | Hybrid search (vector + BM25 + RRF), cross-encoder re-ranking, evaluation with LLM-as-judge | `app/retrieval.py`, `eval/` |

## Run it (Mac)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Terminal 1: the LLM
ollama serve                     # first time: ollama pull llama3.2

# Terminal 2: load documents, then start the API
python -m app.cli_ingest data/sample_docs
python -m uvicorn app.main:app --reload          # API docs: http://localhost:8000/docs

# Terminal 3: the chat UI
streamlit run ui/streamlit_app.py                # http://localhost:8501
```

API examples:
```bash
curl localhost:8000/health
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"session_id": "demo", "message": "How many days of annual leave do I get?"}'
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"session_id": "demo", "message": "and how many can I carry over?"}'
```

## Test and evaluate
```bash
pytest -q                               # 25 tests, offline: fake embeddings + fake LLM
python -m eval.run_eval                 # retrieval: vector vs hybrid (Hit Rate@k, MRR)
python -m eval.run_eval --rerank        # + cross-encoder re-ranking
python -m eval.run_eval --with-llm      # + end-to-end: refusals, citations, faithfulness
```
The bundled sample corpus is tiny, so its scores are not meaningful. Add real documents to
`data/` and questions to `eval/questions.json` before quoting any numbers.

## Docker
```bash
docker compose up --build               # API :8000, UI :8501, Ollama stays on the host
```

## Learn the code
Start with **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)**: every file, the life of a
document and of a question, the concepts behind each step, exercises and interview questions.
