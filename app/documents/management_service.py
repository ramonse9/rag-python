from uuid import UUID

from anyio import to_thread
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import SessionLocal
from app.documents.models import Document, DocumentChunk
from app.documents.schemas import DocumentSummary


class DocumentManagementService:
    """
    Administra los documentos almacenados.

    No necesita OpenAI porque solamente consulta
    y elimina registros de PostgreSQL.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.session_factory = session_factory

    async def list_documents(self) -> list[DocumentSummary]:
        """
        Devuelve todos los documentos y su cantidad de chunks.
        """

        return await to_thread.run_sync(
            self._list_documents,
        )

    async def delete_document(
        self,
        document_id: UUID,
    ) -> bool:
        """
        Elimina un documento.

        Devuelve False cuando el documento no existe.
        """

        return await to_thread.run_sync(
            self._delete_document,
            document_id,
        )

    def _list_documents(self) -> list[DocumentSummary]:
        statement = (
            select(
                Document.id,
                Document.filename,
                Document.created_at,
                func.count(DocumentChunk.id).label("chunks"),
            )
            .outerjoin(
                DocumentChunk,
                DocumentChunk.document_id == Document.id,
            )
            .group_by(
                Document.id,
                Document.filename,
                Document.created_at,
            )
            .order_by(Document.created_at.desc())
        )

        with self.session_factory() as session:
            rows = (
                session.execute(statement)
                .mappings()
                .all()
            )

        return [
            DocumentSummary.model_validate(
                {
                    "id": row["id"],
                    "filename": row["filename"],
                    "createdAt": row["created_at"],
                    "chunks": row["chunks"],
                }
            )
            for row in rows
        ]

    def _delete_document(
        self,
        document_id: UUID,
    ) -> bool:
        with self.session_factory.begin() as session:
            document = session.get(
                Document,
                document_id,
            )

            if document is None:
                return False

            session.delete(document)

        return True


def get_document_management_service(
) -> DocumentManagementService:
    """
    Crea el servicio mediante el sistema de dependencias de FastAPI.
    """

    return DocumentManagementService(
        session_factory=SessionLocal,
    )