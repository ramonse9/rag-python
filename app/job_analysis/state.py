from typing import NotRequired, TypedDict

from app.job_analysis.schemas import (
    JobAnalysisIntent,
    JobRequirement,
    RequirementEvaluation,
    RequirementEvidence,
)


class JobAnalysisState(TypedDict):
    """
    Estado que circula entre los nodos de LangGraph.

    job_description y question son la entrada inicial.
    Los demás campos son agregados progresivamente.
    """

    job_description: str
    question: str

    request_valid: NotRequired[bool]
    validation_reason: NotRequired[str | None]

    requirements: NotRequired[
        list[JobRequirement]
    ]

    evidence: NotRequired[
        list[RequirementEvidence]
    ]

    evaluations: NotRequired[
        list[RequirementEvaluation]
    ]

    intent: NotRequired[JobAnalysisIntent]

    answer: NotRequired[str]