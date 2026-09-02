import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.main import app
from app.documents.management_service import (
    get_document_management_service,
)
from app.documents.schemas import DocumentSummary


DOCUMENT_ID = "9163f8cd-4e9d-4bea-9304-6e7cc63f6abc"
MISSING_ID = "00000000-0000-4000-8000-000000000099"


@pytest.fixture
def management_service(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    service = SimpleNamespace(
        list_documents=AsyncMock(),
        delete_document=AsyncMock(),
    )

    monkeypatch.setitem(
        app.dependency_overrides,
        get_document_management_service,
        lambda: service,
    )

    return service


def request(
    method: str,
    path: str,
) -> httpx.Response:
    """
    Envía una petición a FastAPI sin iniciar Uvicorn.
    """

    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.request(method, path)

    return asyncio.run(send_request())


def test_lists_documents(
    management_service: SimpleNamespace,
) -> None:
    management_service.list_documents.return_value = [
        DocumentSummary.model_validate(
            {
                "id": DOCUMENT_ID,
                "filename": "resume.pdf",
                "createdAt": "2026-09-01T23:40:37",
                "chunks": 11,
            }
        )
    ]

    response = request(
        "GET",
        "/api/documents",
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": DOCUMENT_ID,
            "filename": "resume.pdf",
            "createdAt": "2026-09-01T23:40:37",
            "chunks": 11,
        }
    ]

    management_service.list_documents.assert_awaited_once_with()


def test_deletes_existing_document(
    management_service: SimpleNamespace,
) -> None:
    management_service.delete_document.return_value = True

    response = request(
        "DELETE",
        f"/api/documents/{DOCUMENT_ID}",
    )

    assert response.status_code == 204
    assert response.content == b""

    management_service.delete_document.assert_awaited_once()


def test_returns_404_when_document_does_not_exist(
    management_service: SimpleNamespace,
) -> None:
    management_service.delete_document.return_value = False

    response = request(
        "DELETE",
        f"/api/documents/{MISSING_ID}",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document not found.",
    }

    management_service.delete_document.assert_awaited_once()


def test_rejects_invalid_document_id(
    management_service: SimpleNamespace,
) -> None:
    response = request(
        "DELETE",
        "/api/documents/not-a-uuid",
    )

    assert response.status_code == 422
    management_service.delete_document.assert_not_awaited()