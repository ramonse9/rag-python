from typing import Annotated

from anyio import to_thread
from fastapi import Depends
from sqlalchemy import Float, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import SessionLocal
from app.documents.models import Document, DocumentChunk
from app.documents.schemas import (
    DocumentSearchResult,
    IngestDocumentResponse,
    RagAnswerResponse,
    RagSource,
)
from app.openai.service import (
    OpenAIService,
    get_openai_service,
)

from app.documents.chunking import Chunk, ChunkingService
from app.documents.text_cleaning import TextCleaningService

class DocumentValidationError(ValueError):
    """El documento recibido no contiene datos válidos"""
    pass

class DocumentsService:
    """
    Contiene la lógica de negocio relacionada con documentos.
    """

    def __init__(
        self,
        openai_service: OpenAIService,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.openai_service = openai_service
        self.session_factory = session_factory
        self.text_cleaner = TextCleaningService()
        self.chunker = ChunkingService()

    async def search(
        self,
        query: str,
        top_k: int = 8,
    ) -> list[DocumentSearchResult]:
        """
        Busca los chunks más similares a una consulta.
        """

        embeddings = await self.openai_service.create_embeddings(
            [query]
        )

        query_embedding = embeddings[0]

        return await to_thread.run_sync(
            self._search_by_embedding,
            query_embedding,
            top_k,
        )

    async def ask(
        self,
        question: str,
        top_k: int = 8,
    ) -> RagAnswerResponse:
        """
        Recupera fuentes relevantes y genera una respuesta.
        """

        search_results = await self.search(
            query=question,
            top_k=top_k,
        )

        context = "\n\n".join(
            (
                f"[Source {position}]\n"
                f"{result.content}"
            )
            for position, result in enumerate(
                search_results,
                start=1,
            )
        )

        answer = await self.openai_service.generate_answer(
            question=question,
            context=context,
        )

        sources = [
            RagSource.model_validate(
                {
                    "chunkIndex": result.chunk_index,
                    "distance": result.distance,
                    "filename": result.filename,
                    "content": result.content,
                }
            )
            for result in search_results
        ]

        return RagAnswerResponse(
            question=question,
            answer=answer,
            sources=sources,
        )
    
    async def ingest(
        self,
        filename: str,
        text: str,
    ) -> IngestDocumentResponse:
        """
        Limpia el texto, genera chunks y embeddings,
        y guarda el documento completo.
        """

        if not filename.strip():
            raise DocumentValidationError(
                "The filename cannot be empty."
            )

        cleaned_text = self.text_cleaner.clean(text)
        chunks = self.chunker.split(cleaned_text)

        if not chunks:
            raise DocumentValidationError(
                "The document contains no usable text."
            )

        embeddings = await self.openai_service.create_embeddings(
            [chunk.content for chunk in chunks]
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "The number of embeddings does not match the chunks."
            )

        return await to_thread.run_sync(
            self._save_document,
            filename,
            text,
            chunks,
            embeddings,
        )


    def _save_document(
        self,
        filename: str,
        original_text: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> IngestDocumentResponse:
        """
        Guarda el documento y todos sus chunks
        dentro de una única transacción.
        """

        with self.session_factory.begin() as session:
            document = Document(
                filename=filename,
                original_text=original_text,
            )

            session.add(document)

            # Envía el INSERT sin confirmar todavía la transacción.
            # PostgreSQL genera el UUID y SQLAlchemy lo recupera.
            session.flush()

            chunk_entities = [
                DocumentChunk(
                    content=chunk.content,
                    chunk_index=chunk.index,
                    embedding=embedding,
                    document=document,
                )
                for chunk, embedding in zip(
                    chunks,
                    embeddings,
                    strict=True,
                )
            ]

            session.add_all(chunk_entities)

            result = IngestDocumentResponse.model_validate(
                {
                    "documentId": document.id,
                    "filename": filename,
                    "chunks": len(chunk_entities),
                }
            )

        return result
    
    def _search_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[DocumentSearchResult]:
        """
        Ejecuta la consulta SQL en un worker thread.

        Esto evita bloquear el event loop de FastAPI con
        una sesión SQLAlchemy síncrona.
        """

        distance = DocumentChunk.embedding.op(
            "<=>",
            return_type=Float,
        )(query_embedding).label("distance")

        statement = (
            select(
                DocumentChunk.id,
                DocumentChunk.content,
                DocumentChunk.chunk_index,
                DocumentChunk.document_id,
                Document.filename,
                distance,
            )
            .join(
                Document,
                Document.id == DocumentChunk.document_id,
            )
            .order_by(distance)
            .limit(top_k)
        )

        with self.session_factory() as session:
            rows = (
                session.execute(statement)
                .mappings()
                .all()
            )

        return [
            DocumentSearchResult.model_validate(
                dict(row)
            )
            for row in rows
        ]


def get_documents_service(
    openai_service: Annotated[
        OpenAIService,
        Depends(get_openai_service),
    ],
) -> DocumentsService:
    """
    Construye DocumentsService mediante dependencias FastAPI.
    """

    return DocumentsService(
        openai_service=openai_service,
        session_factory=SessionLocal,
    )
