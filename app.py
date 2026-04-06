import streamlit as st
import tempfile
import os
from ingest import ingest
from agent import ask

st.set_page_config(page_title="RAG Document Q&A", layout="centered")
st.title("RAG Document Q&A")

# --- PDF Upload ---
st.header("1. Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and uploaded_file.name != st.session_state.get("ingested_file"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    with st.spinner(f"Ingesting {uploaded_file.name}..."):
        ingest(tmp_path)
    os.unlink(tmp_path)
    st.session_state["ingested_file"] = uploaded_file.name
    st.success(f"{uploaded_file.name} ingested successfully.")

# --- Question Answering ---
st.header("2. Ask a Question")
question = st.text_input("Enter your question")

if st.button("Ask") and question:
    st.markdown("### Answer")
    response_chunks = []
    with st.spinner("Retrieving..."):
        # Consume first chunk to unblock spinner (retrieval happens before streaming)
        gen = ask(question)
        first = next(gen, None)

    if first is not None:
        def full_stream():
            yield first
            yield from gen

        st.write_stream(full_stream())
