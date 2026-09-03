from pydantic import BaseModel, ConfigDict, Field

from app.job_analysis.schemas import (
    JobAnalysisIntent,
    JobRequirement,
    RequirementMatch,
)


class StructuredOutput(BaseModel):
    """
    Configuración para respuestas estructuradas del modelo.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class RequestValidationResult(StructuredOutput):
    """
    Resultado de validar si la solicitud pertenece
    al flujo de análisis de vacantes.
    """

    valid: bool

    reason: str | None = Field(
        description=(
            "Reason the request is invalid, "
            "or null when it is valid."
        )
    )


class ExtractedRequirementsResult(StructuredOutput):
    """
    Requisitos extraídos de la descripción de empleo.
    """

    requirements: list[JobRequirement]


class RequirementAssessmentResult(StructuredOutput):
    """
    Evaluación generada para un requisito individual.

    El requisito y su importancia se agregan después
    desde los datos originales, evitando que el modelo
    pueda modificarlos.
    """

    match: RequirementMatch

    explanation: str = Field(
        min_length=1,
        description=(
            "Concise explanation based only on "
            "the provided resume evidence."
        ),
    )

    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Short factual statements directly supported "
            "by the resume evidence."
        ),
    )


class IntentClassificationResult(StructuredOutput):
    """
    Clasificación de la intención de la pregunta.
    """

    intent: JobAnalysisIntent