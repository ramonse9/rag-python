import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.main import app
from app.documents.pdf_service import PDFService
from app.documents.router import MAX_PDF_BYTES
from app.documents.schemas import IngestDocumentResponse
from app.documents.service import (
    DocumentAlreadyExistsError,
    DocumentValidationError,
    get_documents_service,
)


ENDPOINT = "/api/rag/ingest"
DOCUMENT_ID = "00000000-0000-4000-8000-000000000001"

UploadedFile = tuple[str, bytes, str]


@pytest.fixture
def ingest_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Sustituye el servicio real por uno simulado."""
    mock = AsyncMock()
    fake_service = SimpleNamespace(ingest=mock)

    monkeypatch.setitem(
        app.dependency_overrides,
        get_documents_service,
        lambda: fake_service,
    )

    return mock


def post_ingest(file: UploadedFile | None = None) -> httpx.Response:
    """Envía una petición a FastAPI sin iniciar un servidor."""
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            files = {"file": file} if file is not None else None

            return await client.post(ENDPOINT, files=files)

    return asyncio.run(send_request())


@pytest.mark.parametrize(
    "file",
    [
        None,
        ("test.txt", b"Hello", "text/plain"),
        ("   ", b"%PDF-1.4", "application/pdf"),
        ("empty.pdf", b"", "application/pdf"),
        ("invalid.pdf", b"not a pdf", "application/pdf"),
    ],
    ids=[
        "missing-file",
        "unsupported-type",
        "blank-filename",
        "empty-file",
        "invalid-pdf",
    ],
)
def test_rejects_invalid_files(
    file: UploadedFile | None,
    ingest_mock: AsyncMock,
) -> None:
    response = post_ingest(file)

    assert response.status_code == 400
    assert "detail" in response.json()
    ingest_mock.assert_not_awaited()


def test_rejects_oversized_file(ingest_mock: AsyncMock) -> None:
    response = post_ingest(
        (
            "large.pdf",
            b"x" * (MAX_PDF_BYTES + 1),
            "application/pdf",
        )
    )

    assert response.status_code == 413
    ingest_mock.assert_not_awaited()


def test_rejects_pdf_without_text(
    ingest_mock: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PDFService,
        "extract_text",
        lambda self, content: "   ",
    )

    response = post_ingest(
        ("test.pdf", b"simulated pdf", "application/pdf")
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Could not extract text from PDF."
    )
    ingest_mock.assert_not_awaited()


def test_returns_400_for_document_validation_error(
    ingest_mock: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PDFService,
        "extract_text",
        lambda self, content: "-- 1 of 2 --",
    )

    ingest_mock.side_effect = DocumentValidationError(
        "The document contains no usable text."
    )

    response = post_ingest(
        ("test.pdf", b"simulated pdf", "application/pdf")
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The document contains no usable text."
    )
    ingest_mock.assert_awaited_once_with(
        filename="test.pdf",
        text="-- 1 of 2 --",
    )

def test_returns_409_when_document_already_exists(
    ingest_mock: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PDFService,
        "extract_text",
        lambda self, content: "Texto de prueba",
    )

    ingest_mock.side_effect = DocumentAlreadyExistsError(
        "A document is already registered. "
        "Delete it before uploading another one."
    )

    response = post_ingest(
        ("test.pdf", b"simulated pdf", "application/pdf")
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": (
            "A document is already registered. "
            "Delete it before uploading another one."
        )
    }

    ingest_mock.assert_awaited_once_with(
        filename="test.pdf",
        text="Texto de prueba",
    )


def test_ingests_pdf_successfully(
    ingest_mock: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PDFService,
        "extract_text",
        lambda self, content: "Texto de prueba",
    )

    ingest_mock.return_value = (
        IngestDocumentResponse.model_validate(
            {
                "documentId": DOCUMENT_ID,
                "filename": "test.pdf",
                "chunks": 2,
            }
        )
    )

    response = post_ingest(
        ("test.pdf", b"simulated pdf", "application/pdf")
    )

    assert response.status_code == 201
    assert response.json() == {
        "documentId": DOCUMENT_ID,
        "filename": "test.pdf",
        "chunks": 2,
    }

    ingest_mock.assert_awaited_once_with(
        filename="test.pdf",
        text="Texto de prueba",
    )