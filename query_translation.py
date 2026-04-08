"""
Query Translation strategies for RAG.

Two approaches implemented here:

1. Multi-Query  — generate N rephrasings, retrieve for each, deduplicate.
2. RAG Fusion   — generate N rephrasings, retrieve ranked lists for each,
                  re-rank with Reciprocal Rank Fusion (RRF), take top-K.

RRF formula:  score(doc) = Σ  1 / (k + rank_i)
              where rank_i is the doc's rank in result list i, and k=60 (constant).
              Docs appearing in multiple lists accumulate higher scores.
"""

import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.errors import ServerError
from ingest import get_vector_store, get_embeddings

load_dotenv()

# ── Prompt that asks the LLM to produce N query variants ─────────────────────

MULTI_QUERY_PROMPT = PromptTemplate(
    template="""You are an AI assistant. Your task is to generate {n} different
versions of the given question to improve document retrieval.
Each version should approach the question from a slightly different angle or
use different wording, so that together they cover more relevant documents.

Original question: {question}

Output ONLY the {n} questions, one per line, no numbering or extra text.""",
    input_variables=["question", "n"],
)

# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

def _generate_queries(question: str, n: int = 3, retries: int = 3) -> list[str]:
    """Use the LLM to rephrase the original question into n variants.
    Retries up to `retries` times on 503 overload errors."""
    chain = MULTI_QUERY_PROMPT | _get_llm() | StrOutputParser()
    for attempt in range(retries):
        try:
            raw = chain.invoke({"question": question, "n": n})
            queries = [q.strip() for q in raw.strip().splitlines() if q.strip()]
            return queries[:n]
        except ServerError as e:
            if "503" in str(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
            else:
                raise

# ── Strategy 1: Multi-Query ───────────────────────────────────────────────────

def multi_query_retrieve(question: str, n: int = 3, k: int = 4) -> list:
    """
    1. Generate n query variants with the LLM.
    2. Retrieve k docs per variant.
    3. Deduplicate by page_content — return the union.
    """
    vs = get_vector_store()
    queries = _generate_queries(question, n)

    seen, docs = set(), []
    for q in queries:
        for doc in vs.similarity_search(q, k=k):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)

    return docs, queries   # return queries so the UI can show them

# ── Strategy 2: RAG Fusion ────────────────────────────────────────────────────

def rag_fusion_retrieve(question: str, n: int = 3, k: int = 4, rrf_k: int = 60) -> list:
    """
    1. Generate n query variants with the LLM.
    2. Retrieve k docs per variant (each call returns a *ranked* list).
    3. Apply Reciprocal Rank Fusion across all ranked lists.
       score(doc) = Σ  1 / (rrf_k + rank)   for every list where doc appears.
    4. Return docs sorted by descending RRF score.

    rrf_k=60 is the standard default from the original RRF paper.
    """
    vs = get_vector_store()
    queries = _generate_queries(question, n)

    # Collect ranked result lists
    ranked_lists: list[list] = []
    for q in queries:
        ranked_lists.append(vs.similarity_search(q, k=k))

    # Compute RRF scores
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}   # content → doc object

    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            key = doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    # Sort by score descending, return top-k docs
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    fused_docs = [doc_map[key] for key in sorted_keys[:k]]

    return fused_docs, queries, {k: round(v, 4) for k, v in rrf_scores.items()}
