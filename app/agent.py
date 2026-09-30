"""Phase 2: LangGraph conversational RAG agent.

A LangGraph app is a state machine: a shared STATE dict flows through NODES (Python
functions that return updates to the state), connected by EDGES (fixed or conditional).

    START -> route --chitchat--> chitchat -> END
               |
               +--documents--> rewrite -> retrieve --found--------> generate -> END
                                             ^   |--nothing, retry--> reformulate --+
                                             |                                      |
                                             +--------------------------------------+
                                                 |--nothing, no retries left--> refuse -> END

  route       LLM classifies the RAW message: "documents" (needs retrieval) or "chitchat".
              Routing happens first so a greeting is never rewritten into a document question.
  rewrite     turns a follow-up ("what about sick leave?") into a standalone query using history
  retrieve    HybridRetriever.search(): returns [] when nothing is relevant
  reformulate LLM rewrites the query with different wording, then we search again
  generate    grounded answer with [n] citations
  refuse      fixed NOT_FOUND answer, no LLM call

Memory: a checkpointer saves the state after every turn under a thread_id. We use the
session_id as thread_id, so each browser session has its own conversation history.
"""
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import Settings
from app.rag import NOT_FOUND, generate_answer, to_sources


class AgentState(TypedDict):
    # add_messages is a "reducer": new messages are APPENDED to the list instead of
    # replacing it. That is how history accumulates across turns.
    messages: Annotated[list[AnyMessage], add_messages]
    question: str          # latest user message
    search_query: str      # standalone query actually used for retrieval
    route: str             # "documents" or "chitchat"
    docs: list[Document]   # retrieved chunks for this turn
    attempts: int          # how many searches this turn
    answer: str
    sources: list[dict]


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest question as a standalone search query, using the chat "
               "history only to resolve references like 'it' or 'what about'. If the question "
               "does not refer to the history, return it unchanged. "
               "Return only the query, nothing else."),
    ("human", "Chat history:\n{history}\n\nLatest question: {question}"),
])

ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You route messages for a company-document assistant. Reply with exactly one word:\n"
               "chitchat  - greetings, thanks, or questions about what the assistant can do\n"
               "documents - anything else (any question that might be answered by company documents)"),
    ("human", "{question}"),
])

REFORMULATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "A document search returned nothing relevant. Rewrite the query using different "
               "wording, synonyms and key terms likely to appear in a policy document. "
               "Return only the new query."),
    ("human", "{query}"),
])

CHITCHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant that answers questions about the user's uploaded "
               "company documents. Reply briefly (1-2 sentences). Do not answer factual questions "
               "from your own knowledge."),
    ("human", "{question}"),
])


def _history_text(messages: list[AnyMessage], limit: int) -> str:
    past = messages[:-1][-limit:]  # everything except the current question, last `limit` only
    return "\n".join(f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" for m in past)


def build_agent(retriever, llm: BaseChatModel, s: Settings, checkpointer=None):
    text = StrOutputParser()  # turns the model's message object into a plain string

    # ---- nodes: each receives the state and returns ONLY the keys it changes ----------
    def route(state: AgentState) -> dict:
        question = state["messages"][-1].content
        label = (ROUTE_PROMPT | llm | text).invoke({"question": question}).lower()
        # Default to documents: wrongly skipping retrieval is worse than one extra search
        return {"question": question, "search_query": question,
                "route": "chitchat" if "chitchat" in label else "documents"}

    def rewrite(state: AgentState) -> dict:
        history = _history_text(state["messages"], s.max_history_messages)
        if not history:  # first turn: nothing to resolve, save an LLM call
            return {}
        query = (REWRITE_PROMPT | llm | text).invoke({"history": history, "question": state["question"]})
        return {"search_query": query.strip() or state["question"]}

    def retrieve(state: AgentState) -> dict:
        docs = retriever.search(state["search_query"], s.top_k)
        return {"docs": docs, "attempts": state.get("attempts", 0) + 1}

    def reformulate(state: AgentState) -> dict:
        query = (REFORMULATE_PROMPT | llm | text).invoke({"query": state["search_query"]})
        return {"search_query": query.strip() or state["search_query"]}

    def generate(state: AgentState) -> dict:
        answer = generate_answer(llm, state["search_query"], state["docs"])
        # The model itself may refuse (layer 2 after the retrieval gate): then show no sources,
        # because listing chunks next to "I could not find this" would be misleading.
        sources = [] if NOT_FOUND.lower() in answer.lower() else to_sources(state["docs"])
        return {"answer": answer, "sources": sources, "messages": [AIMessage(answer)]}

    def chitchat(state: AgentState) -> dict:
        answer = (CHITCHAT_PROMPT | llm | text).invoke({"question": state["question"]})
        return {"answer": answer, "sources": [], "messages": [AIMessage(answer)]}

    def refuse(state: AgentState) -> dict:
        return {"answer": NOT_FOUND, "sources": [], "messages": [AIMessage(NOT_FOUND)]}

    # ---- conditional edges: return the NAME of the next node --------------------------
    def after_route(state: AgentState) -> Literal["rewrite", "chitchat"]:
        return "chitchat" if state["route"] == "chitchat" else "rewrite"

    def after_retrieve(state: AgentState) -> Literal["generate", "reformulate", "refuse"]:
        if state["docs"]:
            return "generate"
        return "reformulate" if state["attempts"] <= s.max_retrieval_retries else "refuse"

    # ---- wire the graph ----------------------------------------------------------------
    g = StateGraph(AgentState)
    for name, fn in [("rewrite", rewrite), ("route", route), ("retrieve", retrieve),
                     ("reformulate", reformulate), ("generate", generate),
                     ("chitchat", chitchat), ("refuse", refuse)]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route)
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve)
    g.add_edge("reformulate", "retrieve")      # the retry loop
    for terminal in ("generate", "chitchat", "refuse"):
        g.add_edge(terminal, END)
    # InMemorySaver keeps history in RAM (lost on restart). In production you would pass
    # a persistent checkpointer (e.g. Postgres or Redis) here instead.
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def run_turn(agent, session_id: str, message: str) -> dict:
    """One chat turn. Per-turn fields are reset; `messages` persists via the checkpointer."""
    result = agent.invoke(
        {"messages": [HumanMessage(message)], "attempts": 0, "docs": [], "sources": [],
         "answer": "", "route": "", "question": "", "search_query": ""},
        config={"configurable": {"thread_id": session_id}},
    )
    return {"answer": result["answer"], "sources": result["sources"],
            "route": result["route"], "search_query": result["search_query"]}
