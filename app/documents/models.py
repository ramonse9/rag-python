from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.base import Base


class Document(Base):
    """A source document whose text is divided into searchable chunks."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    original_text: Mapped[str | None] = mapped_column(
        "originalText",
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        "createdAt",
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentChunk(Base):
    """A fragment of a document and its 1536-dimensional embedding."""

    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(
        "chunkIndex",
        Integer,
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(1536),
        nullable=False,
    )
    document_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,
    )

    document: Mapped[Document | None] = relationship(
        back_populates="chunks",
    )
