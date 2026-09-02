import asyncio
from unittest.mock import AsyncMock
from uuid import UUID

from app.documents.retrievers.postgres import PostgresRetriever
from app.documents.schemas import DocumentSearchResult
from app.documents.service import DocumentsService


DOCUMENT_ID = UUID(
    "9163f8cd-4e9d-4bea-9304-6e7cc63f6abc"
)
CHUNK_ID = UUID(
    "00000000-0000-4000-8000-000000000010"
)


def test_retriever_converts_search_results_to_documents() -> None:
    documents_service = AsyncMock(
        spec=DocumentsService,
    )

    documents_service.search.return_value = [
        DocumentSearchResult.model_validate(
            {
                "id": CHUNK_ID,
                "content": "Experience building APIs with NestJS.",
                "chunkIndex": 3,
                "document_id": DOCUMENT_ID,
                "filename": "resume.pdf",
                "distance": 0.25,
            }
        )
    ]

    retriever = PostgresRetriever(
        documents_service=documents_service,
        top_k=3,
    )

    documents = asyncio.run(
        retriever.ainvoke("What experience does Ramon have?")
    )

    assert len(documents) == 1

    document = documents[0]

    assert document.page_content == (
        "Experience building APIs with NestJS."
    )
    assert document.metadata == {
        "chunkIndex": 3,
        "distance": 0.25,
        "filename": "resume.pdf",
        "documentId": str(DOCUMENT_ID),
    }

    documents_service.search.assert_awaited_once_with(
        query="What experience does Ramon have?",
        top_k=3,
    )