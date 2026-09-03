import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.job_analysis.schemas import JobAnalysisResponse
from app.job_analysis.service import get_job_analysis_service
from app.main import app


ENDPOINT = "/api/job-analysis/analyze"


@pytest.fixture
def job_analysis_service(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    service = SimpleNamespace(
        analyze=AsyncMock(),
    )

    monkeypatch.setitem(
        app.dependency_overrides,
        get_job_analysis_service,
        lambda: service,
    )

    return service


def post_analysis(
    payload: dict[str, object],
) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.post(
                ENDPOINT,
                json=payload,
            )

    return asyncio.run(send_request())


def test_job_analysis_endpoint_returns_complete_response(
    job_analysis_service: SimpleNamespace,
) -> None:
    job_analysis_service.analyze.return_value = (
        JobAnalysisResponse.model_validate(
            {
                "jobDescription": "Backend developer position.",
                "question": "How well does Ramon match?",
                "requestValid": True,
                "requirements": [
                    {
                        "requirement": "NestJS",
                        "category": "backend",
                        "importance": "required",
                    }
                ],
                "evidence": [],
                "evaluations": [
                    {
                        "requirement": "NestJS",
                        "importance": "required",
                        "match": "strong",
                        "explanation": "Direct professional experience.",
                        "evidence": ["Built production NestJS APIs."],
                    }
                ],
                "intent": "match",
                "answer": "The calculated alignment is 100%.",
            }
        )
    )

    response = post_analysis(
        {
            "jobDescription": "Backend developer position.",
            "question": "How well does Ramon match?",
        }
    )

    assert response.status_code == 200
    assert response.json() == {
        "jobDescription": "Backend developer position.",
        "question": "How well does Ramon match?",
        "requestValid": True,
        "requirements": [
            {
                "requirement": "NestJS",
                "category": "backend",
                "importance": "required",
            }
        ],
        "evidence": [],
        "evaluations": [
            {
                "requirement": "NestJS",
                "importance": "required",
                "match": "strong",
                "explanation": "Direct professional experience.",
                "evidence": ["Built production NestJS APIs."],
            }
        ],
        "intent": "match",
        "answer": "The calculated alignment is 100%.",
    }

    job_analysis_service.analyze.assert_awaited_once_with(
        job_description="Backend developer position.",
        question="How well does Ramon match?",
    )


def test_job_analysis_endpoint_omits_none_fields(
    job_analysis_service: SimpleNamespace,
) -> None:
    reason = "The question is unrelated to candidate evaluation."

    job_analysis_service.analyze.return_value = (
        JobAnalysisResponse.model_validate(
            {
                "jobDescription": "Backend developer position.",
                "question": "What will the weather be tomorrow?",
                "requestValid": False,
                "validationReason": reason,
                "requirements": [],
                "evidence": [],
                "evaluations": [],
                "answer": reason,
            }
        )
    )

    response = post_analysis(
        {
            "jobDescription": "Backend developer position.",
            "question": "What will the weather be tomorrow?",
        }
    )

    assert response.status_code == 200
    assert response.json() == {
        "jobDescription": "Backend developer position.",
        "question": "What will the weather be tomorrow?",
        "requestValid": False,
        "validationReason": reason,
        "requirements": [],
        "evidence": [],
        "evaluations": [],
        "answer": reason,
    }
    assert "intent" not in response.json()


@pytest.mark.parametrize(
    "payload",
    [
        {
            "jobDescription": "   ",
            "question": "How well does Ramon match?",
        },
        {
            "jobDescription": "Backend developer position.",
            "question": "   ",
        },
    ],
    ids=[
        "empty-job-description",
        "empty-question",
    ],
)
def test_job_analysis_endpoint_rejects_empty_input(
    payload: dict[str, object],
    job_analysis_service: SimpleNamespace,
) -> None:
    response = post_analysis(payload)

    assert response.status_code == 422
    job_analysis_service.analyze.assert_not_awaited()
