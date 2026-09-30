# Walkthrough: understanding every part of this project

Read this with the code open next to it. Each section says which file to look at.

---

## 1. The problem RAG solves

An LLM only knows what was in its training data. It does not know *your* company's
policies, and when asked it will often invent a confident answer (hallucination).

**Retrieval-Augmented Generation (RAG)** fixes this in two steps:
1. **Retrieve**: find the few passages of your documents that are relevant to the question.
2. **Generate**: give those passages to the LLM and instruct it to answer *only* from them.

The LLM becomes a reader of your documents instead of a memory of the internet.

---

## 2. Project map

| File | Responsibility |
|---|---|
| `app/config.py` | All settings in one place, overridable from `.env` |
| `app/providers.py` | Creates embeddings, vector store, LLM, re-ranker (the only vendor-specific code) |
| `app/ingest.py` | Load files, split into chunks, give each chunk a stable ID, store it |
| `app/retrieval.py` | Vector search, BM25 keyword search, fusion, re-ranking, relevance gate |
| `app/rag.py` | The grounded prompt, context formatting, single-shot answer |
| `app/agent.py` | LangGraph agent: rewrite, route, retrieve, retry, generate, memory |
| `app/main.py` | FastAPI endpoints, startup, error handling |
| `app/cli_ingest.py` | Load a whole folder from the terminal |
| `ui/streamlit_app.py` | Chat interface that calls the API |
| `eval/run_eval.py`, `eval/judge.py` | Measure retrieval and answer quality |
| `tests/` | Offline tests with fake embeddings and fake LLMs |

---

## 3. Life of a document (ingestion)

Entry points: `POST /ingest` in `app/main.py`, or `python -m app.cli_ingest`. Logic: `app/ingest.py`.

1. **Load** (`load_file`). A PDF becomes one `Document` per page, so every chunk remembers
   `source` and `page`. That is what makes citations like "leave_policy.md, page 1" possible.
2. **Chunk** (`chunk_documents`). Pages are split into ~800-character chunks.
   - *Why chunk at all?* Embeddings of a whole document are too blurry to match specific
     questions, and whole documents don't fit the prompt.
   - *Why 800?* Big enough to hold a complete rule with its context, small enough to be precise.
   - *Why 120 overlap?* A sentence cut at a boundary still appears whole in one of the two chunks.
   - `RecursiveCharacterTextSplitter` tries to split at paragraphs first, then lines, then
     sentences, then words, so chunks break at natural boundaries.
3. **ID** (`chunk_id`). SHA-256 of source + page + text. Uploading the same file twice
   produces the same IDs, so Chroma *overwrites* instead of duplicating. This is called
   **idempotent** ingestion.
4. **Embed + store**. `vectorstore.add_documents` runs each chunk through the embedding
   model (`all-MiniLM-L6-v2`, 384 numbers per chunk) and saves vector + text + metadata in
   ChromaDB on disk (`chroma_db/`).
5. **Keyword index**. The API calls `retriever.refresh()`, which rebuilds the in-memory BM25
   index from all stored chunks.

---

## 4. Life of a question (`POST /chat`)

`app/main.py: chat()` → `app/agent.py: run_turn()` → the graph runs:

1. **route**: the LLM classifies the *raw* message with one word, `chitchat` or `documents`.
   Greetings go to a short friendly reply; everything else continues to retrieval. Unclear
   output defaults to `documents`, because skipping retrieval by mistake is the worse error.
   Routing runs *before* rewriting: an earlier version rewrote first, and "hello" after a
   leave question was turned into a leave question (see the regression test
   `test_greeting_after_document_question_is_not_rewritten`).
2. **rewrite**: if the session has history, the LLM rewrites a follow-up into a standalone query.
   "and how many can I carry over?" becomes "How many annual leave days can be carried over?"
   Without this, retrieval would search for an ambiguous fragment. First message: skipped.
3. **retrieve**: `HybridRetriever.search()` (section 5). Returns `[]` if nothing is relevant.
4. **after_retrieve** decides:
   - chunks found → **generate**
   - nothing, retries left → **reformulate** (LLM tries synonyms), then back to retrieve
   - nothing, no retries left → **refuse** with the fixed sentence (no LLM call)
5. **generate**: fills the prompt in `app/rag.py` with numbered chunks and calls the LLM.
   If the model itself answers with the refusal sentence, the sources are dropped. This is
   the second safety layer: e.g. "Who is the CEO of Northwind Bank?" passes the retrieval
   gate (every document mentions Northwind Bank) but the documents don't name a CEO.
6. The answer is appended to `messages`. The **checkpointer** saves the state under
   `thread_id = session_id`, so the next message in the same session sees this history.
   A different `session_id` starts with a clean history.

`POST /query` is the simpler Phase 1 path: retrieve → generate, no memory, no agent.

---

## 5. Retrieval in depth (`app/retrieval.py`)

### Embeddings and cosine similarity
An embedding model maps text to a vector so that texts with similar *meaning* end up close
together. Closeness is measured by **cosine similarity**: 1 = same direction, 0 = unrelated.
"time off" and "annual leave" score high even though they share no words.

### BM25 keyword search
BM25 scores a chunk higher when it contains the query's words, especially **rare** words
(a word in every chunk tells you nothing), with diminishing returns for repeats and a
penalty for very long chunks. It catches exact terms embeddings blur: "EUR 500",
"Security Operations Centre", product codes, names.

### Reciprocal Rank Fusion (RRF)
We have two ranked lists with incomparable scores (cosine 0..1, BM25 0..~20). RRF ignores
the scores and uses only ranks: `score = Σ 1 / (60 + rank)`.

Worked example (k = 60):
| chunk | vector rank | BM25 rank | RRF score |
|---|---|---|---|
| A | 1 | 3 | 1/61 + 1/63 = **0.0323** |
| C | – | 1 | 1/61 = 0.0164 |
| B | 2 | – | 1/62 = 0.0161 |

A wins because *both* methods liked it. That is the point of fusion.

### Cross-encoder re-ranking (optional, `USE_RERANKER=true`)
The embedding model is a **bi-encoder**: question and chunk are embedded separately, then
compared. Fast, but approximate. A **cross-encoder** reads question and chunk *together*,
so it can see that "Can I fly business class?" is answered by "Business class may be booked
for flights of 6 hours or more". Much more accurate, but too slow for every chunk, so it
only re-scores the ~20 fused candidates. Its score also decides relevance.

The model outputs raw **logits** (roughly -12 to +12). We convert them to probabilities
with a sigmoid, `1 / (1 + e^-x)`, ourselves, because whether the library does it depends on
its version. We found this out the hard way: on the sample docs, thresholds 0.3 and 0.1 gave
identical results, and printing the scores showed values like 6.93 and -11.25. After the
sigmoid, relevant chunks scored ≥ 0.16 and unrelated ones ≈ 0.00001, so 0.1 separates them.

### What our experiments showed (sample corpus, 22 questions)
| Configuration | Hit Rate@4 | MRR | Gate |
|---|---|---|---|
| chunks 800, cosine threshold 0.3 | 83% | 0.833 | 75% |
| chunks 300, cosine threshold 0.3 | 94% | 0.944 | 75% |
| chunks 300, cosine threshold 0.2 | 100% | 1.000 | 75% |
| chunks 300, re-ranker on raw logits (bug) | 83% | 0.833 | 100% |

Lessons: chunk size mattered most; the cosine threshold trades recall against rejection;
the re-ranker is the best junk filter once its scores are on the right scale. The corpus is
too small for these percentages to generalise; the trends are what count.

### The relevance gate
Without a re-ranker: if even the best cosine score is below `RELEVANCE_THRESHOLD`, the
question is treated as not covered by the documents and `[]` is returned. BM25 cannot make
this decision because common words match something in almost every corpus.

---

## 6. Prompt engineering (`app/rag.py`, `app/agent.py`)

| Rule in the prompt | Failure it prevents |
|---|---|
| "Answer ONLY from the context" | Using training knowledge (hallucination) |
| Fixed refusal sentence | Vague non-answers; gives code a string to detect |
| "Cite every factual sentence [n]" | Untraceable claims |
| Numbered context with source and page | Gives the model the exact tags to cite |
| `temperature = 0` | Random variation in factual answers |
| Router answers "exactly one word" | Unparseable routing output |

The agent prompts are small, single-purpose instructions. One job per LLM call is more
reliable than one giant prompt, especially with small local models.

---

## 7. LangGraph concepts (`app/agent.py`)

- **State**: a `TypedDict` shared by all nodes. Nodes return only the keys they change.
- **Reducer**: `Annotated[list, add_messages]` means returned messages are *appended*.
  Other keys are simply overwritten.
- **Nodes**: plain Python functions. **Edges**: fixed (`add_edge`) or conditional
  (`add_conditional_edges`, a function returning the next node's name).
- **Cycles**: `reformulate → retrieve` is a loop, bounded by `attempts`. This is what makes
  it an *agent* rather than a fixed chain.
- **Checkpointer**: saves state per `thread_id`. `InMemorySaver` is lost on restart; in
  production you would use a database-backed checkpointer.

---

## 8. The API (`app/main.py`)

- **lifespan**: models load once at startup, not per request.
- **Pydantic models** validate input (`question` must be 3–1000 characters) and generate
  the Swagger docs at `/docs`. Invalid input returns 422 automatically.
- **Error handling**: any LLM failure (Ollama down, bad key, quota) becomes a 503 with a
  clear message instead of a 500 with a stack trace.
- **Security detail**: `Path(f.filename).name` strips paths like `../../etc/passwd` from
  uploaded file names.

---

## 9. Testing (`tests/`)

Tests never call a real model: they are fast, free, and deterministic.
- `KeywordEmbedding` (conftest.py) embeds text as a bag of hashed words, so texts that share
  words really are similar. That lets retrieval logic be tested for real without downloads.
- `FakeListChatModel` returns scripted answers in order, so each test controls exactly what
  the "LLM" says at each graph step.
- FastAPI's `TestClient` calls endpoints in-process, with `get_llm` patched to the fake.

---

## 10. Evaluation (`eval/`)

| Metric | Meaning |
|---|---|
| Hit Rate@k | % of questions where the right document is in the top k |
| MRR | average of 1/rank of the right document (1.0 = always first) |
| Gate | % of out-of-scope questions where retrieval returned nothing |
| Correct refusals | out-of-scope questions answered with the refusal sentence |
| False refusals | in-scope questions wrongly refused |
| Citation rate | answers that contain [n] tags |
| Faithfulness | % of answers the LLM judge found fully supported by the context |

**LLM-as-judge caveats**: the judge is an LLM and can be wrong; small models are weak
judges; the sample corpus is too small for meaningful numbers. To get numbers you can put
on a resume: add 10–20 real public PDFs, write ~40 questions, run all three commands, and
check a sample of judge verdicts by hand.

---

## 11. Suggested learning path

1. **Day 1**: run everything, read sections 1–4, run `pytest -q -v`, read each test.
2. **Day 2**: in `/docs`, call `/chat` with follow-ups; watch `search_query` and `route` change.
3. **Day 3**: change `CHUNK_SIZE` (300 vs 800 vs 1500), re-ingest (delete `chroma_db/` first),
   run the eval, and explain the difference.
4. **Day 4**: compare `RETRIEVAL_MODE=vector` and `hybrid` on keyword-heavy questions.
5. **Day 5**: enable `USE_RERANKER=true` and tune `RERANK_THRESHOLD` using the eval.
6. **Day 6**: add your own PDFs and 40 questions; record real metrics.
7. **Day 7**: break things on purpose (stop Ollama, upload a .docx, send a 1-character
   question) and trace what happens in the code.

---

## 12. Interview questions you should be able to answer

1. *Why RAG instead of fine-tuning?* Knowledge changes: RAG updates by re-ingesting,
   fine-tuning needs retraining. RAG also gives citations and is far cheaper.
2. *How do you reduce hallucination?* Grounded prompt, explicit refusal, relevance gate
   before generation, citations, faithfulness evaluation.
3. *Why hybrid search?* Semantic search misses exact terms; keyword search misses synonyms.
4. *Why RRF instead of adding scores?* The scores are on different scales; ranks are comparable.
5. *Bi-encoder vs cross-encoder?* Separate vs joint encoding; speed vs accuracy; use both in stages.
6. *How does conversation memory work?* Checkpointer keyed by session; history is used to
   rewrite follow-ups into standalone queries before retrieval.
7. *What happens if the LLM provider is down?* 503 with a clear message; the provider is
   swappable via config.
8. *How do you know it works?* Offline tests for logic; eval metrics for quality; judge
   verdicts spot-checked by hand.
9. *What would you change for production?* Persistent checkpointer (Postgres/Redis),
   a vector DB server instead of embedded Chroma, auth, rate limiting, streaming responses,
   tracing (e.g. LangSmith), async ingestion queue for large files.

---

## 13. Known limitations (be honest about these)

- BM25 index and chat memory live in RAM: fine for one server, not for several.
- Chroma runs embedded: don't ingest from the CLI while the API is running.
- PDF text extraction fails on scanned images (would need OCR).
- Small local LLMs sometimes ignore the citation rule or misroute; the eval shows how often.

---

## 14. Working with a real corpus (RBI Master Directions)

Why RBI documents: long, dense regulatory PDFs are exactly what enterprise RAG systems
face, and small wording differences matter ("at least once in every two years").

Workflow:
1. `python -m scripts.download_docs` downloads the PDFs listed in `scripts/download_docs.py`
   into `data/rbi/`.
2. Point the app at a separate index so the sample docs don't mix in: in `.env` set
   `CHROMA_DIR=./chroma_rbi`, then `python -m app.cli_ingest data/rbi`.
3. `python -m scripts.check_questions eval/rbi_questions.json data/rbi` confirms every
   expected answer passage really exists in the extracted PDF text.
4. Tune on the dev split only: `python -m eval.run_eval --questions eval/rbi_questions.json --split dev`
5. When the configuration is final, run the test split once and report that result.

Why `expected_text`: with a 100-page document, "the right file is in the top 4" is almost
always true and says little. The eval now requires a retrieved chunk to contain the passage
that answers the question. A limitation: if chunking splits the passage across two chunks,
a correct retrieval can be scored as a miss; keep expected phrases short.

Why a dev/test split: if you tune thresholds on the same questions you report, the number
is optimistic (overfitting to the test set). Tuning on dev and reporting test is the same
discipline as training/validation/test splits in machine learning.

Questions are written in plain user language, not copied from the document, so keyword
search can't simply match the wording; that makes the comparison between vector, hybrid
and re-ranked retrieval meaningful.
