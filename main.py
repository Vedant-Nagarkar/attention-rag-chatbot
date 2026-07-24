import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ingest import load_chunks
from src.indexing import run_indexing_pipeline
from src.retrieval import run_retrieval_pipeline, get_groq_llm
from src.generation import run_generation_pipeline
from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

app = FastAPI(
    title="Attention Is All You Need — RAG Chatbot API",
    description="Ask questions about the Transformer paper (Vaswani et al., 2017) via hybrid RAG.",
    version="1.0.0",
)

# ── Initialize pipeline once at startup (same as app.py did) ──
logger.info("Initializing RAG pipeline...")
chunks = load_chunks()
bm25, collection = run_indexing_pipeline()
llm = get_groq_llm()
logger.info("Pipeline ready")


class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    chunk_id: int
    rerank_score: float
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    is_fallback: bool
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        retrieved, confident = run_retrieval_pipeline(
            req.question, bm25, chunks, collection, llm
        )
        result = run_generation_pipeline(req.question, retrieved, confident)
        return result

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error processing query")