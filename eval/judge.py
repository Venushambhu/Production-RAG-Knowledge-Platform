"""LLM-as-judge: does the answer contain only claims supported by the retrieved context?

Faithfulness is the key RAG quality metric: an answer can sound right and still include
facts the documents never said (hallucination). A second LLM call checks this.
Limitations to mention in interviews: the judge is itself an LLM, so it can be wrong;
small local models are weaker judges; it should be validated against human labels.
"""
import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You grade answers produced by a document assistant. Decide whether EVERY claim in "
               "the answer is supported by the context. Ignore citation tags like [1]. "
               'Respond with JSON only: {{"faithful": true or false, "reason": "<one sentence>"}}'),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}"),
])


def parse_verdict(raw: str) -> tuple[bool | None, str]:
    """Robust parsing: small models often wrap JSON in text or code fences."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return bool(data.get("faithful")), str(data.get("reason", ""))
        except json.JSONDecodeError:
            pass
    lowered = raw.lower()
    if re.search(r'"?faithful"?\s*:\s*false', lowered):
        return False, raw[:200]
    if re.search(r'"?faithful"?\s*:\s*true', lowered):
        return True, raw[:200]
    return None, f"unparseable: {raw[:200]}"


def judge_faithfulness(llm, context: str, question: str, answer: str) -> tuple[bool | None, str]:
    raw = (JUDGE_PROMPT | llm | StrOutputParser()).invoke(
        {"context": context, "question": question, "answer": answer})
    return parse_verdict(raw)
