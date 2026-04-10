"""
app.py
------
Streamlit demo UI for the Clinical RAG Q&A system.
Run: streamlit run src/ui/app.py
"""

import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Clinical RAG Q&A",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Clinical RAG Q&A System")
st.caption("HIPAA-conscious LLM-powered search over de-identified clinical notes.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    k_docs = st.slider("Documents to retrieve (k)", 1, 10, 5)
    show_sources = st.checkbox("Show source documents", value=True)
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        requests.post(f"{API_URL}/api/v1/reset")
        st.success("Conversation cleared.")

    st.divider()
    st.markdown("**Disclaimer:** This tool surfaces information from de-identified notes only. All clinical decisions must involve a licensed provider.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(f"**Source {i+1}** — `{src['source']}` | {src['note_type']} | {src['encounter_date']}")
                    st.code(src["content"], language=None)

# Input
if prompt := st.chat_input("Ask a clinical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning..."):
            try:
                resp = requests.post(
                    f"{API_URL}/api/v1/query",
                    json={"question": prompt},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                sources = data.get("source_docs", [])
                latency = data.get("latency_ms", 0)

                st.markdown(answer)
                st.caption(f"⏱ {latency}ms | 📄 {len(sources)} sources")

                if sources and show_sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Source {i+1}** — `{src['source']}` | {src['note_type']} | {src['encounter_date']}")
                            st.code(src["content"], language=None)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                st.error(f"Error: {e}")
