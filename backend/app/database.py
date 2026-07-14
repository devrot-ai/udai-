"""SQLAlchemy models + session handling (SQLite by default)."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)
    status = Column(String, default="queued")  # queued|processing|completed|failed
    error = Column(Text, nullable=True)

    doc_type = Column(String, nullable=True)
    doc_type_confidence = Column(Float, nullable=True)
    num_pages = Column(Integer, nullable=True)
    is_scanned = Column(Integer, default=0)
    processing_seconds = Column(Float, nullable=True)

    result_json = Column(Text, nullable=True)  # full extraction result

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
