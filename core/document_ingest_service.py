import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import List

from docx import Document as DocxDocument
from fastapi.concurrency import run_in_threadpool
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import CHUNK_OVERLAP, CHUNK_SIZE, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
from .document_db import Document, DocumentChunk, SessionLocal
from .models import embeddings

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\S+")


def normalize_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.replace("\x00", " ")
    cleaned = cleaned.replace("\ufeff", " ")
    cleaned = cleaned.replace("\u200b", " ").replace("\u200c", " ").replace("\u200d", " ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def read_document_content(path: str, extension: str) -> str:
    extension = extension.lower()
    if extension not in _ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {extension}")

    if extension == ".pdf":
        reader = PdfReader(path)
        page_texts = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(page_texts)

    if extension == ".docx":
        doc = DocxDocument(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]

        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                if any(row_cells):
                    paragraphs.append(" | ".join(row_cells))

        return "\n".join(paragraphs)

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def chunk_text_by_tokens(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if overlap < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []

    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        piece = " ".join(tokens[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end >= len(tokens):
            break

    return chunks


def _ensure_qdrant_collection(client: QdrantClient, vector_size: int) -> None:
    if not client.collection_exists(collection_name=QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def process_document_ingest(document_id: str) -> None:
    db = SessionLocal()
    document = db.query(Document).filter(Document.id == document_id).first()

    if document is None:
        db.close()
        logger.error("Document not found for ingest: %s", document_id)
        return

    try:
        document.status = "processing"
        document.error_message = None
        db.commit()

        _, extension = os.path.splitext(document.stored_name)
        raw_text = read_document_content(document.path, extension)
        normalized = normalize_text(raw_text)
        chunks = chunk_text_by_tokens(normalized, CHUNK_SIZE, CHUNK_OVERLAP)

        if not chunks:
            raise ValueError("Document has no readable content after normalization.")

        if not QDRANT_URL:
            raise ValueError("QDRANT_URL is required for ingest.")

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        vectors = embeddings.embed_documents(chunks)

        if not vectors or not vectors[0]:
            raise ValueError("Failed to create embeddings for chunks.")

        _ensure_qdrant_collection(client, len(vectors[0]))

        created_at = datetime.now(timezone.utc).isoformat()
        points: List[PointStruct] = []
        db_chunk_rows: List[DocumentChunk] = []

        for index, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            payload = {
                "document_id": document.id,
                "filename": document.original_name,
                "stored_name": document.stored_name,
                "path": document.path,
                "chunk_index": index,
                "created_at": created_at,
                "content": chunk_text,
            }

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            db_chunk_rows.append(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=index,
                    content_preview=chunk_text[:200],
                    qdrant_point_id=point_id,
                )
            )

        client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)

        db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()
        db.bulk_save_objects(db_chunk_rows)

        document.total_chunks = len(chunks)
        document.status = "done"
        db.commit()

        logger.info("Document ingest success. document_id=%s total_chunks=%s", document.id, len(chunks))
    except Exception as error:
        db.rollback()

        failed_doc = db.query(Document).filter(Document.id == document_id).first()
        if failed_doc is not None:
            failed_doc.status = "failed"
            failed_doc.error_message = str(error)
            db.commit()

        logger.exception("Document ingest failed. document_id=%s", document_id)
    finally:
        db.close()


async def run_document_ingest_task(document_id: str) -> None:
    # Heavy ingest work runs in threadpool to keep event loop responsive.
    await run_in_threadpool(process_document_ingest, document_id)
