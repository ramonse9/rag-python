from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.documents.router import router as documents_router

from app.documents.management_router import (
    router as document_management_router,
)

from app.job_analysis.router import (
    router as job_analysis_router,
)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    documents_router,
    prefix=settings.api_prefix,
)

app.include_router(
    document_management_router,
    prefix=settings.api_prefix,
)

app.include_router(
    job_analysis_router,
    prefix=settings.api_prefix,
)


@app.get(f"{settings.api_prefix}/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}