import os

# ── Base Paths ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

# ── PDF Source ───────────────────────────────────────────────
PDF_FILENAME = "attention_is_all_you_need.pdf"
PDF_PATH     = os.path.join(RAW_DIR, PDF_FILENAME)

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# ── Embedding Model ──────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── ChromaDB ─────────────────────────────────────────────────
CHROMA_DIR        = os.path.join(MODELS_DIR, "chroma_store")
CHROMA_COLLECTION = "attention_paper"

# ── Retrieval ────────────────────────────────────────────────
BM25_TOP_K       = 10
SEMANTIC_TOP_K   = 10
RRF_K            = 60
RERANK_TOP_N     = 3
CONFIDENCE_THRESHOLD = 0.3

# ── Generation ───────────────────────────────────────────────
GROQ_MODEL       = "llama3-70b-8192"
MAX_TOKENS       = 1024
TEMPERATURE      = 0.2

# ── HyDE ─────────────────────────────────────────────────────
HYDE_MIN_QUERY_WORDS = 4   # skip HyDE for queries shorter than this

# ── RAGAS Eval ───────────────────────────────────────────────
GOLDEN_SET_PATH = os.path.join(DATA_DIR, "golden_test_set.json")
RAGAS_RESULTS_PATH = os.path.join(DATA_DIR, "ragas_results.json")