"""Streamlit chat UI for the RAG assistant. Talks to the FastAPI backend over HTTP.

Run:  streamlit run ui/streamlit_app.py   (with the API running on API_URL)
"""
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Assistant", page_icon="📄")
st.title("📄 Document Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []  # list of dicts: role, content, sources, search_query

# ---- Sidebar: status, upload, reset -------------------------------------------------------
with st.sidebar:
    st.header("Knowledge base")
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API online · {h['chunks_indexed']} chunks")
        st.caption(f"LLM: {h['llm_provider']} · retrieval: {h['retrieval_mode']}"
                   f"{' + re-ranker' if h.get('reranker') else ''}")
    except requests.RequestException:
        st.error(f"API not reachable at {API_URL}")

    uploads = st.file_uploader("Add documents", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if uploads and st.button("Ingest"):
        files = [("files", (f.name, f.getvalue())) for f in uploads]
        r = requests.post(f"{API_URL}/ingest", files=files, timeout=300)
        if r.ok:
            st.success(f"Ingested {r.json()['chunks']} chunks")
        else:
            st.error(r.json().get("detail", r.text))

    if st.button("New conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()


def show_sources(sources: list[dict], search_query: str | None) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        if search_query:
            st.caption(f"Searched for: {search_query}")
        for s in sources:
            scores = " · ".join(f"{label} {s[key]}" for key, label in
                                [("vector_score", "cosine"), ("bm25_score", "BM25"), ("rerank_score", "rerank")]
                                if s.get(key) is not None)
            st.markdown(f"**[{s['id']}] {s['source']}**, page {s['page']}  \n`{scores}`")
            st.caption(s["snippet"] + "…")


# ---- Chat ------------------------------------------------------------------------------------
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            show_sources(turn.get("sources", []), turn.get("search_query"))

if prompt := st.chat_input("Ask about your documents…"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                r = requests.post(f"{API_URL}/chat", timeout=180,
                                  json={"session_id": st.session_state.session_id, "message": prompt})
                data = r.json() if r.ok else {"answer": f"⚠️ {r.json().get('detail', r.text)}", "sources": []}
            except requests.RequestException as exc:
                data = {"answer": f"⚠️ Could not reach the API: {exc}", "sources": []}
        st.markdown(data["answer"])
        show_sources(data.get("sources", []), data.get("search_query"))

    st.session_state.history.append({"role": "assistant", "content": data["answer"],
                                     "sources": data.get("sources", []),
                                     "search_query": data.get("search_query")})
