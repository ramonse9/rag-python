from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document as LangChainDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.documents.service import DocumentsService


class PostgresRetriever(BaseRetriever):
    """
    Adapta DocumentsService al formato de retriever de LangChain.

    La búsqueda vectorial continúa realizándose en PostgreSQL.
    """

    documents_service: DocumentsService
    top_k: int = 8

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[LangChainDocument]:
        """
        LangChain requiere implementar la versión síncrona,
        pero nuestro backend utiliza operaciones asíncronas.
        """

        raise NotImplementedError(
            "Use the asynchronous retriever with ainvoke()."
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[LangChainDocument]:
        """
        Recupera chunks desde PostgreSQL y los convierte
        en documentos de LangChain.
        """

        results = await self.documents_service.search(
            query=query,
            top_k=self.top_k,
        )

        return [
            LangChainDocument(
                page_content=result.content,
                metadata={
                    "chunkIndex": result.chunk_index,
                    "distance": result.distance,
                    "filename": result.filename,
                    "documentId": str(result.document_id),
                },
            )
            for result in results
        ]