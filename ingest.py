import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb

load_dotenv()
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "rag_documents"

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

def ingest(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Delete the collection via the client API — safe even when DB file is open
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection didn't exist yet

    Chroma.from_documents(
        chunks,
        get_embeddings(),
        client=client,
        collection_name=COLLECTION_NAME,
    )
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")

def get_vector_store() -> Chroma:
    """Open the existing vector store for reading."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=get_embeddings())

if __name__ == "__main__":
    import sys
    ingest(sys.argv[1])
