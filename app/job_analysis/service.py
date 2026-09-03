from typing import Annotated, cast

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph

from app.core.config import get_settings
from app.documents.service import (
    DocumentsService,
    get_documents_service,
)
from app.job_analysis.graph import JobAnalysisGraph
from app.job_analysis.nodes import JobAnalysisNodes
from app.job_analysis.schemas import JobAnalysisResponse
from app.job_analysis.state import JobAnalysisState


class JobAnalysisService:
    """
    Ejecuta el grafo completo de análisis de vacantes.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
    ) -> None:
        self.graph = graph

    async def analyze(
        self,
        job_description: str,
        question: str,
    ) -> JobAnalysisResponse:
        """
        Ejecuta LangGraph y transforma el estado final
        en la respuesta esperada por React.
        """

        normalized_job_description = (
            job_description.strip()
        )
        normalized_question = question.strip()

        if not normalized_job_description:
            raise ValueError(
                "The job description cannot be empty."
            )

        if not normalized_question:
            raise ValueError(
                "The question cannot be empty."
            )

        initial_state: JobAnalysisState = {
            "job_description": (
                normalized_job_description
            ),
            "question": normalized_question,
        }

        graph_result = await self.graph.ainvoke(
            initial_state,
        )

        final_state = cast(
            JobAnalysisState,
            graph_result,
        )

        return JobAnalysisResponse.model_validate(
            {
                "jobDescription": (
                    final_state["job_description"]
                ),
                "question": final_state["question"],
                "requestValid": final_state.get(
                    "request_valid",
                    False,
                ),
                "validationReason": final_state.get(
                    "validation_reason",
                ),
                "requirements": final_state.get(
                    "requirements",
                    [],
                ),
                "evidence": final_state.get(
                    "evidence",
                    [],
                ),
                "evaluations": final_state.get(
                    "evaluations",
                    [],
                ),
                "intent": final_state.get(
                    "intent",
                ),
                "answer": final_state.get(
                    "answer",
                    "",
                ),
            }
        )


def get_job_analysis_service(
    documents_service: Annotated[
        DocumentsService,
        Depends(get_documents_service),
    ],
) -> JobAnalysisService:
    """
    Construye los nodos, compila el grafo y crea el servicio.
    """

    nodes = JobAnalysisNodes(
        settings=get_settings(),
        documents_service=documents_service,
    )

    graph = JobAnalysisGraph(
        nodes=nodes,
    ).create()

    return JobAnalysisService(
        graph=graph,
    )