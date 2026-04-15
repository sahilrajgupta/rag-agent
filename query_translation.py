"""
Query Translation strategies for RAG.

Three approaches implemented here:

1. Multi-Query    — generate N rephrasings, retrieve for each, deduplicate.
2. RAG Fusion     — generate N rephrasings, retrieve ranked lists for each,
                    re-rank with Reciprocal Rank Fusion (RRF), take top-K.
3. Decomposition  — break a complex question into simple sub-questions,
                    answer each independently with RAG, then synthesize.
                    Two variants:
                    - Parallel:    all sub-questions answered independently
                    - Sequential:  each sub-answer feeds into the next question

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

# ── Strategy 3: Decomposition ─────────────────────────────────────────────────

DECOMPOSE_PROMPT = PromptTemplate(
    template="""You are an AI assistant. Break the following complex question into
{n} simpler sub-questions that together cover all aspects needed to answer it.
Each sub-question should be self-contained and answerable independently.

Complex question: {question}

Output ONLY the {n} sub-questions, one per line, no numbering or extra text.""",
    input_variables=["question", "n"],
)

SUB_ANSWER_PROMPT = PromptTemplate(
    template="""Answer the question below using ONLY the provided context.
If the context doesn't contain enough information, say "Not found in document."

Context:
{context}

Question: {question}

Answer concisely:""",
    input_variables=["context", "question"],
)

PARALLEL_SYNTHESIS_PROMPT = PromptTemplate(
    template="""You are a helpful assistant. A complex question was broken into
sub-questions. Each sub-question was answered independently using document retrieval.
Synthesize all sub-answers into one coherent final answer to the original question.

Original question: {question}

Sub-questions and answers:
{sub_qa}

Final answer:""",
    input_variables=["question", "sub_qa"],
)

SEQUENTIAL_SYNTHESIS_PROMPT = PromptTemplate(
    template="""You are a helpful assistant. A complex question was answered
step by step, where each step built on the previous answers.
Synthesize the step-by-step reasoning into one coherent final answer.

Original question: {question}

Step-by-step answers:
{steps}

Final answer:""",
    input_variables=["question", "steps"],
)


def _retrieve_and_answer(sub_question: str, k: int = 3) -> tuple[str, list]:
    """Retrieve docs for a sub-question and produce a short answer."""
    vs = get_vector_store()
    docs = vs.similarity_search(sub_question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = SUB_ANSWER_PROMPT | _get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": sub_question})
    return answer, docs


def decomposition_parallel(question: str, n: int = 3) -> dict:
    """
    Parallel decomposition:
    1. Break question into n sub-questions.
    2. Answer each sub-question independently via RAG.
    3. Synthesize all sub-answers into a final answer.

    All sub-questions are answered in isolation — no sub-answer influences another.
    Good for: multi-part questions where parts are independent.
    e.g. "What are the causes AND effects of X?" → answer causes, answer effects, combine.
    """
    llm = _get_llm()

    # Step 1: decompose
    sub_questions = (DECOMPOSE_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "n": n}
    )
    sub_questions = [q.strip() for q in sub_questions.strip().splitlines() if q.strip()][:n]

    # Step 2: answer each independently
    steps = []
    all_docs = []
    for sub_q in sub_questions:
        answer, docs = _retrieve_and_answer(sub_q)
        steps.append({"question": sub_q, "answer": answer, "docs": docs})
        all_docs.extend(docs)

    # Step 3: synthesize
    sub_qa_text = "\n\n".join(
        f"Q: {s['question']}\nA: {s['answer']}" for s in steps
    )
    final = (PARALLEL_SYNTHESIS_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "sub_qa": sub_qa_text}
    )

    return {"final_answer": final, "steps": steps, "all_docs": all_docs}


def decomposition_sequential(question: str, n: int = 3) -> dict:
    """
    Sequential decomposition:
    1. Break question into n sub-questions.
    2. Answer sub-question 1 via RAG.
    3. Answer sub-question 2 using its own RAG retrieval AND the answer from step 2.
    4. Repeat — each step accumulates prior answers as context.
    5. Synthesize the chain of reasoning into a final answer.

    Good for: questions that require building up understanding step by step.
    e.g. "How did X lead to Y which caused Z?" — needs sequential reasoning.
    """
    llm = _get_llm()

    # Step 1: decompose
    sub_questions = (DECOMPOSE_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "n": n}
    )
    sub_questions = [q.strip() for q in sub_questions.strip().splitlines() if q.strip()][:n]

    # Step 2-4: answer sequentially, passing prior answers as extra context
    steps = []
    all_docs = []
    prior_answers = ""

    for sub_q in sub_questions:
        vs = get_vector_store()
        docs = vs.similarity_search(sub_q, k=3)
        all_docs.extend(docs)
        doc_context = "\n\n".join(doc.page_content for doc in docs)

        # Inject prior answers so each step can build on what was learned
        full_context = doc_context
        if prior_answers:
            full_context = f"Previously established:\n{prior_answers}\n\nDocument context:\n{doc_context}"

        chain = SUB_ANSWER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": full_context, "question": sub_q})
        steps.append({"question": sub_q, "answer": answer, "docs": docs})
        prior_answers += f"- {sub_q}: {answer}\n"

    # Step 5: synthesize
    steps_text = "\n\n".join(
        f"Step {i+1} — Q: {s['question']}\nA: {s['answer']}" for i, s in enumerate(steps)
    )
    final = (SEQUENTIAL_SYNTHESIS_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "steps": steps_text}
    )

    return {"final_answer": final, "steps": steps, "all_docs": all_docs}
