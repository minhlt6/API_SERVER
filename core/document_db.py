import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import DOCUMENTS_DATABASE_URL

Base = declarative_base()

_connect_args = {"check_same_thread": False} if DOCUMENTS_DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DOCUMENTS_DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    original_name = Column(String(512), nullable=False)
    stored_name = Column(String(512), nullable=False)
    path = Column(String(1024), nullable=False)
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


def init_document_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_document_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
