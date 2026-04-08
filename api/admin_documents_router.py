import os
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from core.config import MAX_UPLOAD_SIZE_MB, UPLOAD_DIR
from core.document_db import Document, get_document_db
from core.document_ingest_service import run_document_ingest_task

router = APIRouter(prefix="/admin/documents", tags=["admin-documents"])

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class FileTooLargeError(Exception):
    pass


def _save_upload_file_stream(file_obj: Any, destination: str, max_size_bytes: int) -> int:
    total_size = 0
    chunk_size = 1024 * 1024

    with open(destination, "wb") as output:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break

            total_size += len(chunk)
            if total_size > max_size_bytes:
                raise FileTooLargeError("Uploaded file exceeds configured maximum size.")

            output.write(chunk)

    return total_size


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_document_db),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: .pdf, .docx, .txt")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    stored_name = f"{uuid.uuid4()}{extension}"
    stored_path = os.path.abspath(os.path.join(UPLOAD_DIR, stored_name))
    max_size_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    try:
        file.file.seek(0)
        size = await run_in_threadpool(
            _save_upload_file_stream,
            file.file,
            stored_path,
            max_size_bytes,
        )
    except FileTooLargeError:
        if os.path.exists(stored_path):
            os.remove(stored_path)
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Max allowed size is {MAX_UPLOAD_SIZE_MB} MB.",
        )
    except Exception as error:
        if os.path.exists(stored_path):
            os.remove(stored_path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {error}")
    finally:
        await file.close()

    document = Document(
        original_name=file.filename,
        stored_name=stored_name,
        path=stored_path,
        mime_type=file.content_type or "application/octet-stream",
        size=size,
        status="pending",
        total_chunks=0,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    background_tasks.add_task(run_document_ingest_task, document.id)

    return {
        "status": "success",
        "document_id": document.id,
        "original_name": document.original_name,
        "stored_name": document.stored_name,
        "path": document.path,
    }


@router.get("/status/{document_id}")
def get_document_status(document_id: str, db: Session = Depends(get_document_db)) -> Dict[str, Any]:
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "status": "success",
        "document_id": document.id,
        "processing_status": document.status,
        "total_chunks": document.total_chunks,
        "error_message": document.error_message,
        "created_at": document.created_at,
    }


@router.get("")
def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_document_db),
) -> Dict[str, Any]:
    records = (
        db.query(Document)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "status": "success",
        "items": [
            {
                "id": doc.id,
                "original_name": doc.original_name,
                "stored_name": doc.stored_name,
                "status": doc.status,
                "total_chunks": doc.total_chunks,
                "created_at": doc.created_at,
            }
            for doc in records
        ],
    }
