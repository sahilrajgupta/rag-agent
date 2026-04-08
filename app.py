import streamlit as st
import tempfile
import os
from ingest import ingest
from agent import ask
from query_translation import multi_query_retrieve, rag_fusion_retrieve
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", layout="centered")
st.title("RAG Document Q&A")

# ── PDF Upload ────────────────────────────────────────────────────────────────
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

# ── Strategy picker ───────────────────────────────────────────────────────────
st.header("2. Choose Retrieval Strategy")
strategy = st.radio(
    "Strategy",
    ["Agentic RAG", "Multi-Query", "RAG Fusion"],
    horizontal=True,
    help=(
        "Agentic RAG: LLM decides when to retrieve | "
        "Multi-Query: rephrase → retrieve → deduplicate | "
        "RAG Fusion: rephrase → retrieve → RRF re-rank"
    )
)

# ── Question ──────────────────────────────────────────────────────────────────
st.header("3. Ask a Question")
question = st.text_input("Enter your question")

ANSWER_PROMPT = PromptTemplate(
    template="""You are a helpful assistant. Answer based ONLY on the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}""",
    input_variables=["context", "question"],
)

def stream_answer_from_docs(docs, question):
    context = "\n\n".join(
        f"[Page {doc.metadata.get('page','?') + 1}] {doc.page_content}" for doc in docs
    )
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    for chunk in chain.stream({"context": context, "question": question}):
        yield chunk

if st.button("Ask") and question:
    st.markdown("### Answer")

    if strategy == "Agentic RAG":
        st.write_stream(ask(question))

    elif strategy == "Multi-Query":
        with st.spinner("Generating query variants & retrieving..."):
            docs, queries = multi_query_retrieve(question)

        with st.expander("Generated queries"):
            for i, q in enumerate(queries, 1):
                st.write(f"{i}. {q}")
        st.write(f"Retrieved **{len(docs)}** unique chunks.")
        st.write_stream(stream_answer_from_docs(docs, question))

        sources = sorted(set(f"Page {d.metadata.get('page','?') + 1}" for d in docs))
        if sources:
            st.markdown("**Sources:** " + ", ".join(sources))

    elif strategy == "RAG Fusion":
        with st.spinner("Generating query variants, retrieving & fusing..."):
            docs, queries, scores = rag_fusion_retrieve(question)

        with st.expander("Generated queries"):
            for i, q in enumerate(queries, 1):
                st.write(f"{i}. {q}")

        with st.expander("RRF scores (higher = appeared in more lists at higher rank)"):
            for content, score in sorted(scores.items(), key=lambda x: -x[1]):
                st.write(f"**{score}** — {content[:120]}...")

        st.write(f"Retrieved **{len(docs)}** fused chunks.")
        st.write_stream(stream_answer_from_docs(docs, question))

        sources = sorted(set(f"Page {d.metadata.get('page','?') + 1}" for d in docs))
        if sources:
            st.markdown("**Sources:** " + ", ".join(sources))
