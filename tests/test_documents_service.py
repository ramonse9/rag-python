import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.database import SessionLocal
from app.documents.service import (
    DocumentAlreadyExistsError,
    DocumentsService,
)
from app.openai.service import OpenAIService


def test_ingest_does_not_call_openai_when_document_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_service = AsyncMock(spec=OpenAIService)

    service = DocumentsService(
        openai_service=openai_service,
        session_factory=SessionLocal,
    )

    # Simulamos que PostgreSQL ya contiene un documento.
    monkeypatch.setattr(
        service,
        "_document_exists",
        lambda: True,
    )

    with pytest.raises(
        DocumentAlreadyExistsError,
        match="A document is already registered",
    ):
        asyncio.run(
            service.ingest(
                filename="new-document.pdf",
                text="Contenido que no debe procesarse.",
            )
        )

    openai_service.create_embeddings.assert_not_awaited()