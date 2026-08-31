from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from typing import Annotated

from app.documents.schemas import (
    AskDocumentsRequest,
    DocumentSearchResult,
    RagAnswerResponse,
    SearchDocumentsRequest,
    IngestDocumentResponse
)

from app.documents.service import (
    DocumentsService,
    get_documents_service,
    DocumentValidationError
)

from anyio import to_thread

from app.documents.pdf_service import (
    PDFExtractionError,
    PDFService,
)

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)

MAX_PDF_BYTES = 10 * 1024 * 1024


@router.post(
    "/ingest",
    response_model=IngestDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Procesar y guardar un documento PDF",
)
async def ingest_document(
    service: Annotated[
        DocumentsService,
        Depends(get_documents_service),
    ],
    file: Annotated[
        UploadFile | None,
        File(description="Documento PDF con texto extraíble"),
    ] = None,
) -> IngestDocumentResponse:
    """
    Recibe un PDF, extrae su texto y guarda sus chunks.
    """

    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PDF file is required.",
        )

    try:
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported.",
            )

        filename = (file.filename or "").strip()

        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The filename cannot be empty.",
            )

        # Lee como máximo el límite más un byte,
        # suficiente para detectar que se excedió.
        content = await file.read(MAX_PDF_BYTES + 1)

        if len(content) > MAX_PDF_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="The PDF file exceeds the 10 MiB limit.",
            )
    finally:
        await file.close()

    pdf_service = PDFService()

    try:
        extracted_text = await to_thread.run_sync(
            pdf_service.extract_text,
            content,
        )
    except PDFExtractionError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    if not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract text from PDF.",
        )

    try:
        return await service.ingest(
            filename=filename,
            text=extracted_text,
        )
    except DocumentValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error


@router.post(
    "/search",
    response_model=list[DocumentSearchResult],
    summary="Buscar fragmentos similares",
)
async def search_documents(
    request: SearchDocumentsRequest,
    service: Annotated[
        DocumentsService,
        Depends(get_documents_service),
    ],
) -> list[DocumentSearchResult]:
    """
    Busca los chunks más similares a una consulta.
    """

    return await service.search(
        query=request.query,
        top_k=request.top_k,
    )


@router.post(
    "/ask",
    response_model=RagAnswerResponse,
    summary="Responder una pregunta mediante RAG",
)
async def ask_documents(
    request: AskDocumentsRequest,
    service: Annotated[
        DocumentsService,
        Depends(get_documents_service),
    ],
) -> RagAnswerResponse:
    """
    Responde una pregunta utilizando los documentos almacenados.
    """

    return await service.ask(
        question=request.question,
        top_k=request.top_k,
    )


@router.post(
    "/langchain/ask",
    response_model=RagAnswerResponse,
    summary="Responder una pregunta mediante LangChain",
)
async def ask_documents_with_langchain(
    request: AskDocumentsRequest,
) -> RagAnswerResponse:
    """
    Variante del endpoint RAG implementada con LangChain.
    """

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="LangChain question answering is not implemented yet.",
    )