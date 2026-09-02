from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    status,
)

from app.documents.management_service import (
    DocumentManagementService,
    get_document_management_service,
)
from app.documents.schemas import DocumentSummary


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.get(
    "",
    response_model=list[DocumentSummary],
    summary="Listar documentos",
)
async def list_documents(
    service: Annotated[
        DocumentManagementService,
        Depends(get_document_management_service),
    ],
) -> list[DocumentSummary]:
    """
    Devuelve los documentos almacenados y su cantidad de chunks.
    """

    return await service.list_documents()


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Eliminar un documento",
)
async def delete_document(
    document_id: UUID,
    service: Annotated[
        DocumentManagementService,
        Depends(get_document_management_service),
    ],
) -> Response:
    """
    Elimina un documento y sus chunks asociados.
    """

    deleted = await service.delete_document(document_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
    )