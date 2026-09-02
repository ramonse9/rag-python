from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from datetime import datetime


class ApiSchema(BaseModel):
    """
    Configuración compartida por los schemas de la API.

    Python utilizará nombres snake_case internamente,
    mientras que la API podrá usar nombres camelCase.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        str_strip_whitespace=True,
    )


class SearchDocumentsRequest(ApiSchema):
    """
    Body recibido por POST /api/rag/search.
    """

    query: str = Field(
        min_length=1,
        description="Texto utilizado para la búsqueda semántica.",
    )

    top_k: int = Field(
        default=8,
        alias="topK",
        ge=1,
        le=20,
        description="Cantidad máxima de chunks que se devolverán.",
    )


class AskDocumentsRequest(ApiSchema):
    """
    Body recibido por POST /api/rag/ask.
    """

    question: str = Field(
        min_length=1,
        description="Pregunta que se responderá usando los documentos.",
    )

    top_k: int = Field(
        default=8,
        alias="topK",
        ge=1,
        le=20,
        description="Cantidad máxima de fuentes utilizadas.",
    )


class IngestDocumentResponse(ApiSchema):
    """
    Respuesta de POST /api/rag/ingest.
    """

    document_id: UUID = Field(alias="documentId")
    filename: str
    chunks: int

class DocumentSummary(ApiSchema):
    """
    Información resumida de un documento almacenado.
    """

    id: UUID
    filename: str
    created_at: datetime = Field(alias="createdAt")
    chunks: int


class DocumentSearchResult(ApiSchema):
    """
    Un chunk encontrado mediante búsqueda vectorial.
    """

    id: UUID
    content: str
    chunk_index: int = Field(alias="chunkIndex")
    document_id: UUID
    filename: str
    distance: float


class RagSource(ApiSchema):
    """
    Fuente incluida en una respuesta RAG.
    """

    chunk_index: int = Field(alias="chunkIndex")
    distance: float
    filename: str
    content: str


class RagAnswerResponse(ApiSchema):
    """
    Respuesta esperada por el frontend React.
    """

    question: str
    answer: str
    sources: list[RagSource]