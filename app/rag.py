"""Prompt + generation helpers shared by the single-shot /query endpoint and the agent.

The prompt is where "prompt engineering" happens. Each rule targets a known LLM failure:
  - "Answer ONLY from the context"      -> stops the model using training knowledge (hallucination)
  - fixed NOT_FOUND sentence            -> a refusal we can detect in code and in evaluation
  - "cite [n]"                          -> every claim is traceable to a chunk the user can check
  - numbered context blocks with source -> gives the model the tags it must cite
"""
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

NOT_FOUND = "I could not find this in the provided documents."

SYSTEM_PROMPT = """You are a document assistant. Answer ONLY from the context below.
Rules:
- If the context does not contain the answer, reply exactly: "{not_found}"
- Cite every factual sentence with its source tag, e.g. [1] or [2].
- Be concise. Do not use outside knowledge.

Context:
{context}"""

# {context} and {question} are filled at run time; .partial() fixes {not_found} now.
PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
).partial(not_found=NOT_FOUND)


def format_context(docs: list[Document]) -> str:
    """[1] (source: leave_policy.md, page 1)\n<chunk text>\n\n[2] ..."""
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source')}, page {d.metadata.get('page')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def to_sources(docs: list[Document]) -> list[dict]:
    """What the API returns about each chunk. The id matches the [n] tag in the answer."""
    return [
        {"id": i, "source": d.metadata.get("source"), "page": d.metadata.get("page"),
         "snippet": d.page_content[:200],
         "vector_score": d.metadata.get("vector_score"),
         "bm25_score": d.metadata.get("bm25_score"),
         "rerank_score": d.metadata.get("rerank_score")}
        for i, d in enumerate(docs, start=1)
    ]


def generate_answer(llm: BaseChatModel, question: str, docs: list[Document]) -> str:
    # LCEL: prompt -> model -> parser. "|" pipes the output of one step into the next.
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": format_context(docs), "question": question})


def answer(retriever, llm: BaseChatModel, question: str, k: int | None = None) -> dict:
    """Single-shot RAG (no history, no agent): retrieve -> generate."""
    docs = retriever.search(question, k)
    if not docs:                       # nothing relevant: refuse without spending an LLM call
        return {"answer": NOT_FOUND, "sources": []}
    text = generate_answer(llm, question, docs)
    sources = [] if NOT_FOUND.lower() in text.lower() else to_sources(docs)
    return {"answer": text, "sources": sources}
