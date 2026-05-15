import json
import os
import re

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PDF_PATH,
    PROCESSED_DIR,
    RAW_DIR,
)
from src.logger import get_logger

logger = get_logger(__name__)


def check_pdf_exists() -> str:
    """
    Checks if the PDF exists in data/raw/.
    Raises a clear error with instructions if not found.

    Returns:
        str: path to the PDF file
    """
    logger.debug(f"check_pdf_exists called — checking: {PDF_PATH}")

    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(PDF_PATH):
        logger.info(f"PDF found at {PDF_PATH}")
        return PDF_PATH

    logger.critical(
        f"PDF not found at {PDF_PATH}. "
        f"Download from https://arxiv.org/pdf/1706.03762 "
        f"and save as attention_is_all_you_need.pdf in data/raw/"
    )
    raise FileNotFoundError(
        f"PDF missing. Download from https://arxiv.org/pdf/1706.03762 "
        f"and place at {PDF_PATH}"
    )


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text from every page of the PDF using PyMuPDF.

    Args:
        pdf_path: absolute path to the PDF

    Returns:
        str: full raw text of the document
    """
    logger.debug(f"extract_text_from_pdf called with pdf_path={pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        full_text = ""

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            full_text += page_text
            logger.debug(f"Page {page_num + 1}: {len(page_text)} characters extracted")

        doc.close()
        logger.info(f"PDF extraction complete — total characters: {len(full_text)}")
        return full_text

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}", exc_info=True)
        raise


def clean_text(raw_text: str) -> str:
    """
    Cleans raw PDF text.
    Fixes three common research paper PDF artifacts:
    - Multiple blank lines from section breaks
    - Double spaces from column layout
    - Hyphenated line breaks (e.g. trans-\nformer -> transformer)

    Args:
        raw_text: raw extracted text

    Returns:
        str: cleaned text
    """
    logger.debug("clean_text called")

    try:
        # Fix hyphenated line breaks first
        text = re.sub(r'-\n(\w)', r'\1', raw_text)

        # Collapse 3+ newlines into 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Collapse multiple spaces into one
        text = re.sub(r' {2,}', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        logger.info(
            f"Text cleaning complete — "
            f"{len(raw_text)} → {len(text)} characters"
        )
        return text

    except Exception as e:
        logger.error(f"Text cleaning failed: {e}", exc_info=True)
        raise


def chunk_text(cleaned_text: str) -> list[dict]:
    """
    Splits cleaned text into overlapping chunks.
    Uses RecursiveCharacterTextSplitter which tries to split
    on paragraph breaks first, then sentences, then words.

    Args:
        cleaned_text: cleaned full document text

    Returns:
        list of dicts with keys: chunk_id, text, char_count
    """
    logger.debug(f"chunk_text called — input length: {len(cleaned_text)}")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        raw_chunks = splitter.split_text(cleaned_text)

        chunks = [
            {
                "chunk_id": i,
                "text": chunk,
                "char_count": len(chunk)
            }
            for i, chunk in enumerate(raw_chunks)
        ]

        avg_size = sum(c["char_count"] for c in chunks) // len(chunks)
        logger.info(f"Chunking complete — {len(chunks)} chunks, avg size: {avg_size} chars")
        return chunks

    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        raise


def save_chunks(chunks: list[dict]) -> str:
    """
    Saves chunks to data/processed/chunks.json.
    Avoids reprocessing the PDF on every run.

    Args:
        chunks: list of chunk dicts

    Returns:
        str: path where chunks were saved
    """
    logger.debug(f"save_chunks called — {len(chunks)} chunks")

    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        output_path = os.path.join(PROCESSED_DIR, "chunks.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Chunks saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to save chunks: {e}", exc_info=True)
        raise


def load_chunks() -> list[dict]:
    """
    Loads previously saved chunks from data/processed/chunks.json.
    Use this on subsequent runs to skip PDF reprocessing.

    Returns:
        list of chunk dicts
    """
    chunks_path = os.path.join(PROCESSED_DIR, "chunks.json")
    logger.debug(f"load_chunks called — path: {chunks_path}")

    try:
        if not os.path.exists(chunks_path):
            logger.warning("No saved chunks found — run ingest pipeline first")
            raise FileNotFoundError(f"No chunks at {chunks_path}")

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
        return chunks

    except Exception as e:
        logger.error(f"Failed to load chunks: {e}", exc_info=True)
        raise


def run_ingest_pipeline() -> list[dict]:
    """
    Master function — runs the full ingestion pipeline end to end.
    This is the only function other modules should call.

    Flow: check PDF → extract text → clean → chunk → save → return

    Returns:
        list of chunk dicts ready for indexing
    """
    logger.info("=== Ingest Pipeline Started ===")

    pdf_path = check_pdf_exists()
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned  = clean_text(raw_text)
    chunks   = chunk_text(cleaned)
    save_chunks(chunks)

    logger.info(f"=== Ingest Pipeline Complete — {len(chunks)} chunks ready ===")
    return chunks