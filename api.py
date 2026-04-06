from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from agent import ask
from ingest import ingest
import shutil, os

app = FastAPI(title="RAG Document Q&A Agent")

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF into the vector store"""
    path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    ingest(path)
    return {"message": f"{file.filename} ingested successfully"}

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    """Ask a question against ingested documents"""
    result = ask(req.question)
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

