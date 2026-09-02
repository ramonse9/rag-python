from typing import Annotated

from fastapi import Depends
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.documents.retrievers.postgres import PostgresRetriever
from app.documents.schemas import (
    RagAnswerResponse,
    RagSource,
)
from app.documents.service import (
    DocumentsService,
    get_documents_service,
)
from app.openai.service import RAG_INSTRUCTIONS


class LangChainDocumentsService:
    """
    Implementación RAG construida mediante LangChain.
    """

    def __init__(
        self,
        documents_service: DocumentsService,
        settings: Settings,
    ) -> None:
        if settings.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY is required to use LangChain."
            )

        self.documents_service = documents_service

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    RAG_INSTRUCTIONS,
                ),
                (
                    "human",
                    (
                        "QUESTION:\n"
                        "{question}\n\n"
                        "CONTEXT:\n"
                        "{context}"
                    ),
                ),
            ]
        )

        self.model = ChatOpenAI(
            model=settings.openai_response_model,
            api_key=settings.openai_api_key,
            use_responses_api=True,
            store=False,
        )

        self.chain = (
            self.prompt
            | self.model
            | StrOutputParser()
        )

    async def ask(
        self,
        question: str,
        top_k: int = 8,
    ) -> RagAnswerResponse:
        """
        Recupera documentos y ejecuta la cadena RAG.
        """

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "The question cannot be empty."
            )

        retriever = PostgresRetriever(
            documents_service=self.documents_service,
            top_k=top_k,
        )

        documents = await retriever.ainvoke(
            normalized_question,
        )

        context = "\n\n".join(
            (
                f"[Source {position}]\n"
                f"{document.page_content}"
            )
            for position, document in enumerate(
                documents,
                start=1,
            )
        )

        answer = await self.chain.ainvoke(
            {
                "question": normalized_question,
                "context": context,
            }
        )

        normalized_answer = answer.strip()

        if not normalized_answer:
            raise RuntimeError(
                "LangChain returned an empty answer."
            )

        sources = [
            RagSource.model_validate(
                {
                    "chunkIndex": document.metadata[
                        "chunkIndex"
                    ],
                    "distance": document.metadata[
                        "distance"
                    ],
                    "filename": document.metadata[
                        "filename"
                    ],
                    "content": document.page_content,
                }
            )
            for document in documents
        ]

        return RagAnswerResponse(
            question=normalized_question,
            answer=normalized_answer,
            sources=sources,
        )


def get_langchain_documents_service(
    documents_service: Annotated[
        DocumentsService,
        Depends(get_documents_service),
    ],
) -> LangChainDocumentsService:
    """
    Construye el servicio LangChain mediante FastAPI.
    """

    return LangChainDocumentsService(
        documents_service=documents_service,
        settings=get_settings(),
    )