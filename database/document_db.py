import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, func, inspect, or_, text
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from core.config import DOCUMENTS_DATABASE_URL

Base = declarative_base()
logger = logging.getLogger(__name__)

_connect_args = {"check_same_thread": False} if DOCUMENTS_DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DOCUMENTS_DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

RETRY_BASE_SECONDS_DEFAULT = 60
RETRY_MAX_SECONDS_DEFAULT = 3600


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    original_name = Column(String(512), nullable=False)
    stored_name = Column(String(512), nullable=False)
    path = Column(String(1024), nullable=False)
    object_path = Column(String(1024), nullable=True, unique=True, index=True)
    folder_key = Column(String(255), nullable=True)
    collection_name = Column(String(255), nullable=True)
    source_etag = Column(String(255), nullable=True)
    source_updated_at = Column(DateTime(timezone=True), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    last_synced_at = Column(DateTime(timezone=True), nullable=True)
    mime_type = Column(String(255), nullable=False)
    size = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    total_chunks = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content_preview = Column(String(200), nullable=False)
    qdrant_point_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    document = relationship("Document", back_populates="chunks")


class DocumentSyncError(Base):
    __tablename__ = "document_sync_errors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    object_path = Column(String(1024), nullable=False, index=True)
    folder_key = Column(String(255), nullable=True, index=True)
    collection_name = Column(String(255), nullable=True)
    operation = Column(String(64), nullable=False, index=True)
    error_message = Column(Text, nullable=False)
    payload_json = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    next_retry_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_error_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    resolved_at = Column(DateTime(timezone=True), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


def _ensure_documents_schema_compatibility() -> None:
    inspector = inspect(engine)
    if not inspector.has_table("documents"):
        return

    existing_columns = {column["name"] for column in inspector.get_columns("documents")}
    ddl_by_column = {
        "object_path": "ALTER TABLE documents ADD COLUMN object_path VARCHAR(1024)",
        "folder_key": "ALTER TABLE documents ADD COLUMN folder_key VARCHAR(255)",
        "collection_name": "ALTER TABLE documents ADD COLUMN collection_name VARCHAR(255)",
        "source_etag": "ALTER TABLE documents ADD COLUMN source_etag VARCHAR(255)",
        "source_updated_at": "ALTER TABLE documents ADD COLUMN source_updated_at TIMESTAMP",
        "deleted_at": "ALTER TABLE documents ADD COLUMN deleted_at TIMESTAMP",
        "last_synced_at": "ALTER TABLE documents ADD COLUMN last_synced_at TIMESTAMP",
    }

    try:
        with engine.begin() as connection:
            for column_name, ddl in ddl_by_column.items():
                if column_name not in existing_columns:
                    connection.execute(text(ddl))

            connection.execute(
                text("CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_object_path ON documents (object_path)")
            )
    except Exception:
        logger.exception("Failed to ensure documents schema compatibility")


def init_document_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_documents_schema_compatibility()


def _compute_next_retry_at(
    retry_count: int,
    base_retry_seconds: int = RETRY_BASE_SECONDS_DEFAULT,
    max_retry_seconds: int = RETRY_MAX_SECONDS_DEFAULT,
) -> datetime:
    safe_retry_count = max(1, int(retry_count))
    safe_base = max(1, int(base_retry_seconds))
    safe_max = max(safe_base, int(max_retry_seconds))

    delay_seconds = min(safe_max, safe_base * (2 ** (safe_retry_count - 1)))
    return utcnow() + timedelta(seconds=delay_seconds)


def log_document_sync_error(
    db: Session,
    *,
    object_path: str,
    operation: str,
    error_message: str,
    folder_key: Optional[str] = None,
    collection_name: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    base_retry_seconds: int = RETRY_BASE_SECONDS_DEFAULT,
    max_retry_seconds: int = RETRY_MAX_SECONDS_DEFAULT,
) -> DocumentSyncError:
    normalized_path = (object_path or "").strip() or "__global__"
    normalized_operation = (operation or "").strip() or "sync"

    row = (
        db.query(DocumentSyncError)
        .filter(
            DocumentSyncError.object_path == normalized_path,
            DocumentSyncError.operation == normalized_operation,
            DocumentSyncError.resolved_at.is_(None),
        )
        .order_by(DocumentSyncError.last_error_at.desc())
        .first()
    )

    if row is None:
        row = DocumentSyncError(
            object_path=normalized_path,
            operation=normalized_operation,
            retry_count=0,
        )
        db.add(row)

    if folder_key is not None:
        row.folder_key = (folder_key or "").strip() or None
    if collection_name is not None:
        row.collection_name = (collection_name or "").strip() or None

    row.error_message = (error_message or "Unknown sync error").strip() or "Unknown sync error"
    row.payload_json = json.dumps(payload, ensure_ascii=False) if payload else None
    row.retry_count = int(row.retry_count or 0) + 1
    row.last_error_at = utcnow()
    row.next_retry_at = _compute_next_retry_at(
        retry_count=row.retry_count,
        base_retry_seconds=base_retry_seconds,
        max_retry_seconds=max_retry_seconds,
    )
    row.resolved_at = None

    db.commit()
    db.refresh(row)
    return row


def list_due_document_sync_errors(
    db: Session,
    limit: int = 100,
    as_of: Optional[datetime] = None,
) -> List[DocumentSyncError]:
    target_time = as_of or utcnow()
    safe_limit = max(1, min(int(limit or 100), 1000))

    return (
        db.query(DocumentSyncError)
        .filter(
            DocumentSyncError.resolved_at.is_(None),
            or_(
                DocumentSyncError.next_retry_at.is_(None),
                DocumentSyncError.next_retry_at <= target_time,
            ),
        )
        .order_by(DocumentSyncError.next_retry_at.asc(), DocumentSyncError.last_error_at.asc())
        .limit(safe_limit)
        .all()
    )


def mark_document_sync_error_resolved(
    db: Session,
    *,
    object_path: str,
    operation: Optional[str] = None,
) -> int:
    normalized_path = (object_path or "").strip() or "__global__"

    query = db.query(DocumentSyncError).filter(
        DocumentSyncError.object_path == normalized_path,
        DocumentSyncError.resolved_at.is_(None),
    )

    normalized_operation = (operation or "").strip()
    if normalized_operation:
        query = query.filter(DocumentSyncError.operation == normalized_operation)

    rows = query.all()
    if not rows:
        return 0

    resolved_time = utcnow()
    for row in rows:
        row.resolved_at = resolved_time

    db.commit()
    return len(rows)


def count_unresolved_document_sync_errors(db: Session) -> int:
    return int(
        db.query(func.count(DocumentSyncError.id))
        .filter(DocumentSyncError.resolved_at.is_(None))
        .scalar()
        or 0
    )


def list_active_collection_names(db: Session, limit: int = 3) -> List[str]:
    safe_limit = max(1, min(int(limit or 3), 100))

    rows = (
        db.query(
            Document.collection_name,
            func.count(Document.id).label("doc_count"),
            func.max(Document.last_synced_at).label("last_sync"),
        )
        .filter(
            Document.collection_name.isnot(None),
            Document.collection_name != "",
            Document.deleted_at.is_(None),
            Document.status == "done",
        )
        .group_by(Document.collection_name)
        .order_by(func.max(Document.last_synced_at).desc(), func.count(Document.id).desc())
        .limit(safe_limit)
        .all()
    )

    return [str(row[0]) for row in rows if row and row[0]]


def get_document_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
