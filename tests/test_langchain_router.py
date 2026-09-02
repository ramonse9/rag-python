import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.main import app
from app.documents.langchain_service import (
    get_langchain_documents_service,
)
from app.documents.schemas import RagAnswerResponse


ENDPOINT = "/api/rag/langchain/ask"


@pytest.fixture
def langchain_service(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    service = SimpleNamespace(
        ask=AsyncMock(),
    )

    monkeypatch.setitem(
        app.dependency_overrides,
        get_langchain_documents_service,
        lambda: service,
    )

    return service


def post_question(
    payload: dict[str, object],
) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.post(
                ENDPOINT,
                json=payload,
            )

    return asyncio.run(send_request())


def test_langchain_endpoint_returns_answer(
    langchain_service: SimpleNamespace,
) -> None:
    langchain_service.ask.return_value = (
        RagAnswerResponse.model_validate(
            {
                "question": (
                    "What experience does Ramon have?"
                ),
                "answer": (
                    "Ramon has experience building APIs."
                ),
                "sources": [
                    {
                        "chunkIndex": 2,
                        "distance": 0.18,
                        "filename": "resume.pdf",
                        "content": (
                            "Experience building APIs."
                        ),
                    }
                ],
            }
        )
    )

    response = post_question(
        {
            "question": (
                "What experience does Ramon have?"
            ),
            "topK": 3,
        }
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "What experience does Ramon have?",
        "answer": "Ramon has experience building APIs.",
        "sources": [
            {
                "chunkIndex": 2,
                "distance": 0.18,
                "filename": "resume.pdf",
                "content": "Experience building APIs.",
            }
        ],
    }

    langchain_service.ask.assert_awaited_once_with(
        question="What experience does Ramon have?",
        top_k=3,
    )


def test_langchain_endpoint_rejects_empty_question(
    langchain_service: SimpleNamespace,
) -> None:
    response = post_question(
        {
            "question": "   ",
        }
    )

    assert response.status_code == 422
    langchain_service.ask.assert_not_awaited()