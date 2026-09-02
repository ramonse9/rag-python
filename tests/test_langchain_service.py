import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.core.config import get_settings
from app.documents.langchain_service import (
    LangChainDocumentsService,
)
from app.documents.schemas import DocumentSearchResult
from app.documents.service import DocumentsService


DOCUMENT_ID = UUID(
    "9163f8cd-4e9d-4bea-9304-6e7cc63f6abc"
)
CHUNK_ID = UUID(
    "00000000-0000-4000-8000-000000000020"
)


def test_langchain_ask_returns_answer_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents_service = AsyncMock(
        spec=DocumentsService,
    )

    documents_service.search.return_value = [
        DocumentSearchResult.model_validate(
            {
                "id": CHUNK_ID,
                "content": (
                    "Ramón has experience building APIs "
                    "with NestJS and PostgreSQL."
                ),
                "chunkIndex": 2,
                "document_id": DOCUMENT_ID,
                "filename": "resume.pdf",
                "distance": 0.18,
            }
        )
    ]

    service = LangChainDocumentsService(
        documents_service=documents_service,
        settings=get_settings(),
    )

    fake_chain = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=(
                "Ramón has experience with NestJS "
                "and PostgreSQL."
            )
        )
    )

    monkeypatch.setattr(
        service,
        "chain",
        fake_chain,
    )

    result = asyncio.run(
        service.ask(
            question=(
                "  What experience does Ramón have "
                "with NestJS?  "
            ),
            top_k=3,
        )
    )

    assert result.model_dump(by_alias=True) == {
        "question": (
            "What experience does Ramón have with NestJS?"
        ),
        "answer": (
            "Ramón has experience with NestJS "
            "and PostgreSQL."
        ),
        "sources": [
            {
                "chunkIndex": 2,
                "distance": 0.18,
                "filename": "resume.pdf",
                "content": (
                    "Ramón has experience building APIs "
                    "with NestJS and PostgreSQL."
                ),
            }
        ],
    }

    documents_service.search.assert_awaited_once_with(
        query=(
            "What experience does Ramón have with NestJS?"
        ),
        top_k=3,
    )

    fake_chain.ainvoke.assert_awaited_once_with(
        {
            "question": (
                "What experience does Ramón have "
                "with NestJS?"
            ),
            "context": (
                "[Source 1]\n"
                "Ramón has experience building APIs "
                "with NestJS and PostgreSQL."
            ),
        }
    )