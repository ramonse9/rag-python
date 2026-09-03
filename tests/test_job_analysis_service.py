import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from langgraph.graph.state import CompiledStateGraph

from app.job_analysis.service import JobAnalysisService


def create_service(
    graph_result: dict[str, object],
) -> tuple[JobAnalysisService, SimpleNamespace]:
    fake_graph = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=graph_result,
        )
    )

    service = JobAnalysisService(
        graph=cast(
            CompiledStateGraph,
            fake_graph,
        ),
    )

    return service, fake_graph


def test_analyze_returns_complete_job_analysis_response() -> None:
    service, fake_graph = create_service(
        {
            "job_description": "Backend developer position.",
            "question": "How well does Ramon match?",
            "request_valid": True,
            "validation_reason": None,
            "requirements": [
                {
                    "requirement": "NestJS",
                    "category": "backend",
                    "importance": "required",
                }
            ],
            "evidence": [
                {
                    "requirement": "NestJS",
                    "evidence": [
                        {
                            "content": "Built production NestJS APIs.",
                            "filename": "resume.pdf",
                            "chunkIndex": 2,
                            "distance": 0.15,
                        }
                    ],
                }
            ],
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

    result = asyncio.run(
        service.analyze(
            job_description="  Backend developer position.  ",
            question="  How well does Ramon match?  ",
        )
    )

    assert result.model_dump(
        by_alias=True,
        exclude_none=True,
    ) == {
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
        "evidence": [
            {
                "requirement": "NestJS",
                "evidence": [
                    {
                        "content": "Built production NestJS APIs.",
                        "filename": "resume.pdf",
                        "chunkIndex": 2,
                        "distance": 0.15,
                    }
                ],
            }
        ],
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

    fake_graph.ainvoke.assert_awaited_once_with(
        {
            "job_description": "Backend developer position.",
            "question": "How well does Ramon match?",
        }
    )


def test_analyze_returns_invalid_workflow_response() -> None:
    reason = "The question is unrelated to candidate evaluation."

    service, _ = create_service(
        {
            "job_description": "Backend developer position.",
            "question": "What will the weather be tomorrow?",
            "request_valid": False,
            "validation_reason": reason,
            "answer": reason,
        }
    )

    result = asyncio.run(
        service.analyze(
            job_description="Backend developer position.",
            question="What will the weather be tomorrow?",
        )
    )

    assert result.model_dump(
        by_alias=True,
        exclude_none=True,
    ) == {
        "jobDescription": "Backend developer position.",
        "question": "What will the weather be tomorrow?",
        "requestValid": False,
        "validationReason": reason,
        "requirements": [],
        "evidence": [],
        "evaluations": [],
        "answer": reason,
    }


@pytest.mark.parametrize(
    ("job_description", "question", "message"),
    [
        (
            "   ",
            "How well does Ramon match?",
            "The job description cannot be empty.",
        ),
        (
            "Backend developer position.",
            "   ",
            "The question cannot be empty.",
        ),
    ],
)
def test_analyze_rejects_empty_inputs(
    job_description: str,
    question: str,
    message: str,
) -> None:
    service, fake_graph = create_service({})

    with pytest.raises(
        ValueError,
        match=message,
    ):
        asyncio.run(
            service.analyze(
                job_description=job_description,
                question=question,
            )
        )

    fake_graph.ainvoke.assert_not_awaited()
