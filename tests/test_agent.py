from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agent import build_agent, run_turn
from app.config import Settings
from app.ingest import ingest_paths
from app.rag import NOT_FOUND
from app.retrieval import HybridRetriever

SETTINGS = Settings(relevance_threshold=0.2, max_retrieval_retries=1, top_k=2, chunk_size=300)


@pytest.fixture
def loaded_store(vectorstore):
    ingest_paths(sorted(Path("data/sample_docs").glob("*.md")), vectorstore, 300, 50)
    return vectorstore


def make_agent(store, responses):
    llm = FakeListChatModel(responses=responses)
    return build_agent(HybridRetriever(store, SETTINGS), llm, SETTINGS), llm


def test_document_question_is_answered_with_sources(loaded_store):
    agent, _ = make_agent(loaded_store, ["documents", "Employees get 24 days [1]."])
    out = run_turn(agent, "s1", "How many days of annual leave do full-time employees get?")
    assert out["route"] == "documents"
    assert out["answer"] == "Employees get 24 days [1]."
    assert out["sources"][0]["source"] == "leave_policy.md"


def test_chitchat_skips_retrieval(loaded_store):
    agent, _ = make_agent(loaded_store, ["chitchat", "Hi! Ask me about your documents."])
    out = run_turn(agent, "s1", "hello there")
    assert out["route"] == "chitchat" and out["sources"] == []


def test_irrelevant_question_is_reformulated_then_refused(loaded_store):
    # route -> documents, retrieval finds nothing, reformulate once, still nothing -> refuse
    agent, llm = make_agent(loaded_store, ["documents", "zebra quantum chromodynamics"])
    out = run_turn(agent, "s1", "zebra quantum chromodynamics")
    assert out["answer"] == NOT_FOUND and out["sources"] == []
    assert out["search_query"] == "zebra quantum chromodynamics"


def test_follow_up_uses_history_and_sessions_are_isolated(loaded_store):
    agent, _ = make_agent(loaded_store, [
        "documents", "24 days [1].",                                    # turn 1: route, generate
        "documents",                                                    # turn 2: route
        "When is a medical certificate needed for sick leave?",         # turn 2: rewrite
        "After 3 working days [1].",                                    # turn 2: generate
        "documents", "After 3 working days [1].",                       # other session: route, generate
    ])
    run_turn(agent, "s1", "How many annual leave days do employees get?")
    follow_up = run_turn(agent, "s1", "and for sick leave, when do I need a certificate?")
    assert follow_up["search_query"] == "When is a medical certificate needed for sick leave?"
    assert follow_up["sources"][0]["source"] == "leave_policy.md"

    # A new session has no history, so the question is not rewritten
    fresh = run_turn(agent, "s2", "and for sick leave, when do I need a certificate?")
    assert fresh["search_query"] == "and for sick leave, when do I need a certificate?"


def test_greeting_after_document_question_is_not_rewritten(loaded_store):
    # Regression: "hello" used to be rewritten into a leave question and answered from documents
    agent, _ = make_agent(loaded_store, [
        "documents", "24 days [1].",                  # turn 1
        "chitchat", "Hi! Ask me about your documents.",  # turn 2: route first, no rewrite call
    ])
    run_turn(agent, "s1", "How many annual leave days do employees get?")
    out = run_turn(agent, "s1", "hello")
    assert out["route"] == "chitchat" and out["sources"] == []
    assert out["answer"] == "Hi! Ask me about your documents."


def test_model_refusal_returns_no_sources(loaded_store):
    # Retrieval finds chunks (the question shares words with the documents), but the LLM refuses
    question = "Who approves annual leave for the CEO of Northwind Bank?"
    assert HybridRetriever(loaded_store, SETTINGS).search(question)  # retrieval DID find chunks
    agent, _ = make_agent(loaded_store, ["documents", NOT_FOUND])
    out = run_turn(agent, "s1", question)
    assert out["answer"] == NOT_FOUND and out["sources"] == []
