from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RequirementImportance = Literal[
    "required",
    "preferred",
    "valuable",
]

RequirementMatch = Literal[
    "strong",
    "partial",
    "gap",
]

JobAnalysisIntent = Literal[
    "match",
    "gaps",
    "strengths",
    "interview",
    "unsupported",
]

JobRequirementCategory = Literal[
    "frontend",
    "backend",
    "language",
    "database",
    "cloud",
    "ai",
    "architecture",
    "devops",
    "communication",
    "experience",
    "other",
]


class JobAnalysisSchema(BaseModel):
    """
    Configuración compartida por los esquemas Job Analysis.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        str_strip_whitespace=True,
    )


class JobAnalysisRequest(JobAnalysisSchema):
    """
    Body de POST /api/job-analysis/analyze.
    """

    job_description: str = Field(
        alias="jobDescription",
        min_length=1,
        description="Descripción de la vacante.",
    )

    question: str = Field(
        min_length=1,
        description="Pregunta sobre el candidato y la vacante.",
    )


class JobRequirement(JobAnalysisSchema):
    """
    Requisito extraído de la descripción de empleo.
    """

    requirement: str = Field(min_length=1)
    category: JobRequirementCategory
    importance: RequirementImportance


class JobEvidenceChunk(JobAnalysisSchema):
    """
    Fragmento del CV relacionado con un requisito.
    """

    content: str
    filename: str = Field(min_length=1)
    chunk_index: int = Field(
        alias="chunkIndex",
        ge=0,
    )
    distance: float


class RequirementEvidence(JobAnalysisSchema):
    """
    Evidencia recuperada para un requisito.
    """

    requirement: str = Field(min_length=1)
    evidence: list[JobEvidenceChunk] = Field(
        default_factory=list,
    )


class RequirementEvaluation(JobAnalysisSchema):
    """
    Evaluación del candidato contra un requisito.
    """

    requirement: str = Field(min_length=1)
    importance: RequirementImportance
    match: RequirementMatch
    explanation: str = Field(min_length=1)
    evidence: list[str] = Field(
        default_factory=list,
    )


class JobAnalysisResponse(JobAnalysisSchema):
    """
    Respuesta completa esperada por React.
    """

    job_description: str = Field(
        alias="jobDescription",
        min_length=1,
    )

    question: str = Field(min_length=1)

    request_valid: bool = Field(
        alias="requestValid",
    )

    validation_reason: str | None = Field(
        default=None,
        alias="validationReason",
    )

    requirements: list[JobRequirement] = Field(
        default_factory=list,
    )

    evidence: list[RequirementEvidence] = Field(
        default_factory=list,
    )

    evaluations: list[RequirementEvaluation] = Field(
        default_factory=list,
    )

    intent: JobAnalysisIntent | None = None

    answer: str = ""