from typing import Literal

from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.graph.state import CompiledStateGraph

from app.job_analysis.nodes import JobAnalysisNodes
from app.job_analysis.schemas import JobAnalysisIntent
from app.job_analysis.state import JobAnalysisState


ValidationRoute = Literal[
    "valid",
    "invalid",
]


class JobAnalysisGraph:
    """
    Construye y conecta el flujo de análisis de vacantes.
    """

    def __init__(
        self,
        nodes: JobAnalysisNodes,
    ) -> None:
        self.nodes = nodes

    @staticmethod
    def route_after_validation(
        state: JobAnalysisState,
    ) -> ValidationRoute:
        """
        Una solicitud inválida termina inmediatamente.
        """

        return (
            "valid"
            if state.get("request_valid") is True
            else "invalid"
        )

    @staticmethod
    def route_after_intent(
        state: JobAnalysisState,
    ) -> JobAnalysisIntent:
        """
        Selecciona el generador correspondiente.

        Si la intención falta, utiliza la ruta segura
        unsupported.
        """

        return state.get(
            "intent",
            "unsupported",
        )

    def create(self) -> CompiledStateGraph:
        """
        Compila el grafo para poder ejecutarlo con ainvoke().
        """

        workflow = StateGraph(
            JobAnalysisState,
        )

        workflow.add_node(
            "validate_request",
            self.nodes.validate_request,
        )

        workflow.add_node(
            "extract_requirements",
            self.nodes.extract_requirements,
        )

        workflow.add_node(
            "retrieve_evidence",
            self.nodes.retrieve_evidence,
        )

        workflow.add_node(
            "evaluate_requirements",
            self.nodes.evaluate_requirements,
        )

        workflow.add_node(
            "classify_intent",
            self.nodes.classify_intent,
        )

        workflow.add_node(
            "generate_match_answer",
            self.nodes.generate_match_answer,
        )

        workflow.add_node(
            "generate_gaps_answer",
            self.nodes.generate_gaps_answer,
        )

        workflow.add_node(
            "generate_strengths_answer",
            self.nodes.generate_strengths_answer,
        )

        workflow.add_node(
            "generate_interview_answer",
            self.nodes.generate_interview_answer,
        )

        workflow.add_node(
            "generate_unsupported_answer",
            self.nodes.generate_unsupported_answer,
        )

        workflow.add_edge(
            START,
            "validate_request",
        )

        workflow.add_conditional_edges(
            "validate_request",
            self.route_after_validation,
            {
                "valid": "extract_requirements",
                "invalid": "generate_unsupported_answer",
            },
        )

        workflow.add_edge(
            "extract_requirements",
            "retrieve_evidence",
        )

        workflow.add_edge(
            "retrieve_evidence",
            "evaluate_requirements",
        )

        workflow.add_edge(
            "evaluate_requirements",
            "classify_intent",
        )

        workflow.add_conditional_edges(
            "classify_intent",
            self.route_after_intent,
            {
                "match": "generate_match_answer",
                "gaps": "generate_gaps_answer",
                "strengths": "generate_strengths_answer",
                "interview": "generate_interview_answer",
                "unsupported": "generate_unsupported_answer",
            },
        )

        workflow.add_edge(
            "generate_match_answer",
            END,
        )

        workflow.add_edge(
            "generate_gaps_answer",
            END,
        )

        workflow.add_edge(
            "generate_strengths_answer",
            END,
        )

        workflow.add_edge(
            "generate_interview_answer",
            END,
        )

        workflow.add_edge(
            "generate_unsupported_answer",
            END,
        )

        return workflow.compile()