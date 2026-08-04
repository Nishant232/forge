from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.llm.schemas import FinalAnswer, Plan, PlanStep


class TestPlanStep:
    def test_valid_construction(self) -> None:
        step = PlanStep(description="Read the file", expected_tools=["read_file"])
        assert step.description == "Read the file"
        assert step.expected_tools == ["read_file"]

    def test_empty_tools_list_allowed(self) -> None:
        step = PlanStep(description="Think", expected_tools=[])
        assert step.expected_tools == []

    def test_missing_description_raises(self) -> None:
        with pytest.raises(ValidationError):
            PlanStep(expected_tools=["read_file"])  # type: ignore[call-arg]

    def test_missing_expected_tools_raises(self) -> None:
        with pytest.raises(ValidationError):
            PlanStep(description="Read file")  # type: ignore[call-arg]

    def test_serialises_to_dict(self) -> None:
        step = PlanStep(description="d", expected_tools=["t"])
        d = step.model_dump()
        assert d == {"description": "d", "expected_tools": ["t"]}

    def test_round_trip_via_json(self) -> None:
        step = PlanStep(description="do x", expected_tools=["a", "b"])
        rehydrated = PlanStep.model_validate_json(step.model_dump_json())
        assert rehydrated == step


class TestPlan:
    def test_valid_plan(self) -> None:
        plan = Plan(
            steps=[PlanStep(description="Read", expected_tools=["read_file"])],
            rationale="Need to read first",
        )
        assert len(plan.steps) == 1
        assert plan.rationale == "Need to read first"

    def test_empty_steps_allowed(self) -> None:
        plan = Plan(steps=[], rationale="Nothing to do")
        assert plan.steps == []

    def test_missing_rationale_raises(self) -> None:
        with pytest.raises(ValidationError):
            Plan(steps=[])  # type: ignore[call-arg]

    def test_missing_steps_raises(self) -> None:
        with pytest.raises(ValidationError):
            Plan(rationale="ok")  # type: ignore[call-arg]

    def test_multiple_steps(self) -> None:
        plan = Plan(
            steps=[
                PlanStep(description="step1", expected_tools=["read_file"]),
                PlanStep(description="step2", expected_tools=["edit_file"]),
                PlanStep(description="step3", expected_tools=["run_tests"]),
            ],
            rationale="Full flow",
        )
        assert len(plan.steps) == 3
        assert plan.steps[1].description == "step2"

    def test_round_trip_json(self) -> None:
        plan = Plan(
            steps=[PlanStep(description="d", expected_tools=["t"])],
            rationale="r",
        )
        reloaded = Plan.model_validate_json(plan.model_dump_json())
        assert reloaded == plan

    def test_from_dict(self) -> None:
        data = {
            "steps": [{"description": "d", "expected_tools": ["t"]}],
            "rationale": "r",
        }
        plan = Plan.model_validate(data)
        assert plan.rationale == "r"


class TestFinalAnswer:
    def test_valid_success(self) -> None:
        fa = FinalAnswer(summary="Fixed bug", outcome="success")
        assert fa.outcome == "success"
        assert fa.files_changed == []

    def test_valid_with_files(self) -> None:
        fa = FinalAnswer(
            summary="Done", outcome="partial", files_changed=["a.py", "b.py"]
        )
        assert fa.files_changed == ["a.py", "b.py"]

    def test_invalid_outcome_raises(self) -> None:
        with pytest.raises(ValidationError):
            FinalAnswer(summary="x", outcome="unknown")  # type: ignore[arg-type]

    def test_all_outcome_values_accepted(self) -> None:
        for outcome in ("success", "partial", "failure"):
            fa = FinalAnswer(summary="x", outcome=outcome)  # type: ignore[arg-type]
            assert fa.outcome == outcome

    def test_missing_summary_raises(self) -> None:
        with pytest.raises(ValidationError):
            FinalAnswer(outcome="success")  # type: ignore[call-arg]

    def test_missing_outcome_raises(self) -> None:
        with pytest.raises(ValidationError):
            FinalAnswer(summary="x")  # type: ignore[call-arg]

    def test_round_trip_json(self) -> None:
        fa = FinalAnswer(
            summary="All done", outcome="failure", files_changed=["x.py"]
        )
        reloaded = FinalAnswer.model_validate_json(fa.model_dump_json())
        assert reloaded == fa
