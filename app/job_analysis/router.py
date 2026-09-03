from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
)

from app.job_analysis.schemas import (
    JobAnalysisRequest,
    JobAnalysisResponse,
)
from app.job_analysis.service import (
    JobAnalysisService,
    get_job_analysis_service,
)


router = APIRouter(
    prefix="/job-analysis",
    tags=["Job Analysis"],
)


@router.post(
    "/analyze",
    response_model=JobAnalysisResponse,
    response_model_exclude_none=True,
    summary="Analizar una vacante contra el CV",
)
async def analyze_job(
    request: JobAnalysisRequest,
    service: Annotated[
        JobAnalysisService,
        Depends(get_job_analysis_service),
    ],
) -> JobAnalysisResponse:
    """
    Compara los requisitos de una vacante con la
    evidencia profesional almacenada.
    """

    return await service.analyze(
        job_description=request.job_description,
        question=request.question,
    )