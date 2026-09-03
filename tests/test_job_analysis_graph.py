import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from app.job_analysis.graph import JobAnalysisGraph
from app.job_analysis.nodes import JobAnalysisNodes
from app.job_analysis.schemas import (
    JobEvidenceChunk,
    JobRequirement,
    RequirementEvaluation,
    RequirementEvidence,
)


def create_fake_nodes() -> tuple[JobAnalysisNodes, SimpleNamespace]:
    fake = SimpleNamespace(
        validate_request=AsyncMock(),
        extract_requirements=AsyncMock(),
        retrieve_evidence=AsyncMock(),
        evaluate_requirements=AsyncMock(),
        classify_intent=AsyncMock(),
        generate_match_answer=AsyncMock(),
        generate_gaps_answer=AsyncMock(),
        generate_strengths_answer=AsyncMock(),
        generate_interview_answer=AsyncMock(),
        generate_unsupported_answer=AsyncMock(),
    )

    return cast(JobAnalysisNodes, fake), fake


def test_graph_runs_complete_valid_match_route() -> None:
    nodes, fake = create_fake_nodes()

    requirement = JobRequirement(
        requirement="NestJS",
        category="backend",
        importance="required",
    )

    evidence = RequirementEvidence(
        requirement="NestJS",
        evidence=[
            JobEvidenceChunk(
                content="Built production NestJS APIs.",
                filename="resume.pdf",
                chunk_index=2,
                distance=0.15,
            )
        ],
    )

    evaluation = RequirementEvaluation(
        requirement="NestJS",
        importance="required",
        match="strong",
        explanation="Direct professional experience.",
        evidence=["Built production NestJS APIs."],
    )

    fake.validate_request.return_value = {
        "request_valid": True,
        "validation_reason": None,
    }
    fake.extract_requirements.return_value = {
        "requirements": [requirement],
    }
    fake.retrieve_evidence.return_value = {
        "evidence": [evidence],
    }
    fake.evaluate_requirements.return_value = {
        "evaluations": [evaluation],
    }
    fake.classify_intent.return_value = {
        "intent": "match",
    }
    fake.generate_match_answer.return_value = {
        "answer": "The calculated alignment is 100%.",
    }

    graph = JobAnalysisGraph(nodes).create()

    result = asyncio.run(
        graph.ainvoke(
            {
                "job_description": (
                    "NestJS experience is required."
                ),
                "question": (
                    "How well does Ramon match?"
                ),
            }
        )
    )

    assert result["request_valid"] is True
    assert result["requirements"] == [requirement]
    assert result["evidence"] == [evidence]
    assert result["evaluations"] == [evaluation]
    assert result["intent"] == "match"
    assert result["answer"] == (
        "The calculated alignment is 100%."
    )

    fake.validate_request.assert_awaited_once()
    fake.extract_requirements.assert_awaited_once()
    fake.retrieve_evidence.assert_awaited_once()
    fake.evaluate_requirements.assert_awaited_once()
    fake.classify_intent.assert_awaited_once()
    fake.generate_match_answer.assert_awaited_once()

    fake.generate_gaps_answer.assert_not_awaited()
    fake.generate_strengths_answer.assert_not_awaited()
    fake.generate_interview_answer.assert_not_awaited()
    fake.generate_unsupported_answer.assert_not_awaited()


def test_graph_short_circuits_invalid_request() -> None:
    nodes, fake = create_fake_nodes()

    reason = (
        "The question is unrelated to candidate evaluation."
    )

    fake.validate_request.return_value = {
        "request_valid": False,
        "validation_reason": reason,
    }
    fake.generate_unsupported_answer.return_value = {
        "answer": reason,
    }

    graph = JobAnalysisGraph(nodes).create()

    result = asyncio.run(
        graph.ainvoke(
            {
                "job_description": "Backend position.",
                "question": "What will the weather be tomorrow?",
            }
        )
    )

    assert result["request_valid"] is False
    assert result["validation_reason"] == reason
    assert result["answer"] == reason

    fake.validate_request.assert_awaited_once()
    fake.generate_unsupported_answer.assert_awaited_once()

    unsupported_state = (
        fake.generate_unsupported_answer.await_args.args[0]
    )
    assert unsupported_state["request_valid"] is False
    assert unsupported_state["validation_reason"] == reason

    fake.extract_requirements.assert_not_awaited()
    fake.retrieve_evidence.assert_not_awaited()
    fake.evaluate_requirements.assert_not_awaited()
    fake.classify_intent.assert_not_awaited()
    fake.generate_match_answer.assert_not_awaited()
    fake.generate_gaps_answer.assert_not_awaited()
    fake.generate_strengths_answer.assert_not_awaited()
    fake.generate_interview_answer.assert_not_awaited()
