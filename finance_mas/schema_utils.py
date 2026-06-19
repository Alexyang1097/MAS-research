from __future__ import annotations

import json
from typing import Any

from .schemas import (
    AggregationResult,
    CritiqueOutput,
    DeciderResult,
    EvidenceItem,
    ExpectedOutput,
    GoalState,
    InspectionResult,
    Plan,
    PlanRubricCheck,
    RubricResult,
    SubagentOutput,
    ValidationResult,
)


def parse_decider_result(data: dict[str, Any]) -> DeciderResult:
    normalized = dict(data)
    normalized["should_answer"] = _coerce_bool(normalized.get("should_answer", False))
    normalized["reason"] = _coerce_str(normalized.get("reason"), "No reason provided.")
    normalized["confidence"] = _clamp_float(normalized.get("confidence", 0.5), 0.0, 1.0)
    if normalized.get("next_goal_state") is not None:
        normalized["next_goal_state"] = _normalize_goal_state(normalized["next_goal_state"])
    return DeciderResult.model_validate(normalized)


def parse_plan(data: dict[str, Any]) -> Plan:
    normalized = dict(data)

    if normalized.get("mode") in {"independent_subtasks", "parallel", "parallel_tasks"}:
        normalized["mode"] = "parallel_subtasks"
    elif normalized.get("mode") not in {"parallel_subtasks", "perspective_debate"}:
        normalized["mode"] = "parallel_subtasks"

    if "tasks" not in normalized and "independent_subtasks" in normalized:
        normalized["tasks"] = normalized.pop("independent_subtasks")

    normalized["goal_state"] = _normalize_goal_state(normalized.get("goal_state") or normalized.get("next_goal"))
    normalized["decomposition_diagnosis"] = _coerce_str(
        normalized.get("decomposition_diagnosis"),
        "Planner did not provide a decomposition diagnosis.",
    )
    normalized["tasks"] = [_normalize_task_assignment(item) for item in _coerce_list(normalized.get("tasks"))]
    normalized["perspectives"] = [
        _normalize_perspective_assignment(item) for item in _coerce_list(normalized.get("perspectives"))
    ]
    normalized["debate_rounds"] = _clamp_int(normalized.get("debate_rounds", 1), 1, 3)
    normalized["aggregation_rule"] = _coerce_str(
        normalized.get("aggregation_rule"),
        "assess_each_task_then_deduplicate_align_evidence_and_resolve_conflicts",
    )
    normalized["ready_for_final"] = _coerce_bool(normalized.get("ready_for_final", False))
    return Plan.model_validate(normalized)


def parse_inspection_result(data: dict[str, Any]) -> InspectionResult:
    normalized = dict(data)
    normalized["passed"] = _coerce_bool(normalized.get("passed", False))
    normalized["score"] = _clamp_float(normalized.get("score", 0.5), 0.0, 1.0)
    for key in ("completeness", "solvability", "assignment_fit", "redundancy"):
        normalized[key] = _normalize_plan_rubric_check(normalized.get(key), default_passed=normalized["passed"])
    normalized["issues"] = _coerce_str_list(normalized.get("issues"))
    normalized["refinements"] = _coerce_str_list(normalized.get("refinements"))
    if normalized.get("refined_plan"):
        normalized["refined_plan"] = parse_plan(normalized["refined_plan"])
    else:
        normalized["refined_plan"] = None
    return InspectionResult.model_validate(normalized)


def parse_subagent_output(data: dict[str, Any], *, agent_name: str, assignment_id: str, goal: str) -> SubagentOutput:
    normalized = dict(data)
    normalized["agent_name"] = _coerce_str(normalized.get("agent_name"), agent_name)
    normalized["assignment_id"] = _coerce_str(normalized.get("assignment_id"), assignment_id)
    normalized["goal"] = _coerce_str(normalized.get("goal"), goal)
    normalized["answer"] = _coerce_str(
        normalized.get("answer") or normalized.get("final_answer") or normalized.get("result") or _structured_answer_fallback(normalized),
        "No substantive answer provided.",
    )
    normalized["evidence"] = [_normalize_evidence_item(item) for item in _coerce_list(normalized.get("evidence"))]
    normalized["calculations"] = _coerce_str_list(normalized.get("calculations"))
    normalized["assumptions"] = _coerce_str_list(normalized.get("assumptions"))
    normalized["open_questions"] = _coerce_str_list(normalized.get("open_questions"))
    normalized["failure_modes"] = _coerce_str_list(normalized.get("failure_modes"))
    normalized["confidence"] = _clamp_float(normalized.get("confidence", 0.5), 0.0, 1.0)
    return SubagentOutput.model_validate(normalized)


def _structured_answer_fallback(raw: dict[str, Any]) -> str | None:
    ignored = {
        "needs_tools",
        "tool_calls",
        "agent_name",
        "assignment_id",
        "goal",
        "evidence",
        "calculations",
        "assumptions",
        "open_questions",
        "failure_modes",
        "confidence",
    }
    content = {key: value for key, value in raw.items() if key not in ignored and value not in (None, "", [], {})}
    if not content:
        return None
    return json_dumps_compact(content)


def parse_critique_output(data: dict[str, Any], *, agent_name: str) -> CritiqueOutput:
    normalized = dict(data)
    normalized["agent_name"] = agent_name
    normalized["target_assignment_ids"] = _coerce_str_list(normalized.get("target_assignment_ids"))
    normalized["agreements"] = _coerce_str_list(normalized.get("agreements"))
    normalized["disagreements"] = _coerce_str_list(normalized.get("disagreements"))
    normalized["evidence_gaps"] = _coerce_str_list(normalized.get("evidence_gaps"))
    normalized["recommended_revision"] = _coerce_str(
        normalized.get("recommended_revision"),
        "No recommended revision provided.",
    )
    normalized["confidence"] = _clamp_float(normalized.get("confidence", 0.5), 0.0, 1.0)
    return CritiqueOutput.model_validate(normalized)


def parse_aggregation_result(data: dict[str, Any]) -> AggregationResult:
    normalized = dict(data)
    normalized["task_assessments"] = [
        _normalize_task_assessment(item) for item in _coerce_list(normalized.get("task_assessments"))
    ]
    normalized["accepted_facts"] = [
        _normalize_evidence_item(item) for item in _coerce_list(normalized.get("accepted_facts"))
    ]
    normalized["duplicate_facts"] = _coerce_str_list(normalized.get("duplicate_facts"))
    normalized["conflicts"] = _coerce_str_list(normalized.get("conflicts"))
    normalized["evidence_alignment_summary"] = _coerce_str(normalized.get("evidence_alignment_summary"), "")
    normalized["confidence_fusion"] = _coerce_str(normalized.get("confidence_fusion"), "")
    candidate = normalized.get("candidate_answer")
    normalized["candidate_answer"] = None if candidate is None else _coerce_str(candidate, "")
    normalized["unresolved_issues"] = _coerce_str_list(normalized.get("unresolved_issues"))
    return AggregationResult.model_validate(normalized)


def parse_validation_result(data: dict[str, Any]) -> ValidationResult:
    normalized = dict(data)
    normalized["passed"] = _coerce_bool(normalized.get("passed", False))
    normalized["score"] = _clamp_float(normalized.get("score", 0.5), 0.0, 1.0)
    normalized["achieved"] = _coerce_str_list(normalized.get("achieved"))
    normalized["missing"] = _coerce_str_list(normalized.get("missing"))
    normalized["concerns"] = _coerce_str_list(normalized.get("concerns"))
    suggested = normalized.get("suggested_next_goal")
    normalized["suggested_next_goal"] = None if suggested is None else _coerce_str(suggested, "")
    updates = []
    for raw_update in _coerce_list(normalized.get("capability_updates")):
        if isinstance(raw_update, dict):
            updates.append(_normalize_capability_update(raw_update))
    normalized["capability_updates"] = updates
    return ValidationResult.model_validate(normalized)


def parse_rubric_result(data: dict[str, Any]) -> RubricResult:
    normalized = dict(data)
    normalized["passed"] = _coerce_bool(normalized.get("passed", False))
    normalized["score"] = _clamp_float(normalized.get("score", 0.5), 0.0, 1.0)
    normalized["numeric_accuracy"] = _coerce_str(normalized.get("numeric_accuracy"), "")
    normalized["logic_correctness"] = _coerce_str(normalized.get("logic_correctness"), "")
    normalized["evidence_quality"] = _coerce_str(normalized.get("evidence_quality"), "")
    normalized["answer_completeness"] = _coerce_str(normalized.get("answer_completeness"), "")
    normalized["required_fixes"] = _coerce_str_list(normalized.get("required_fixes"))
    return RubricResult.model_validate(normalized)


def fallback_subagent_output(*, agent_name: str, assignment_id: str, goal: str, error: Exception) -> SubagentOutput:
    return SubagentOutput(
        agent_name=agent_name,
        assignment_id=assignment_id,
        goal=goal,
        answer=f"Subagent failed to produce schema-valid output: {error}",
        open_questions=[str(error)],
        failure_modes=["schema_validation_failure"],
        confidence=0.0,
    )


def _normalize_goal_state(raw: Any) -> dict[str, Any]:
    if isinstance(raw, GoalState):
        return raw.model_dump(mode="json")
    if isinstance(raw, dict):
        return {
            "goal": _coerce_str(raw.get("goal") or raw.get("description"), "Continue progress toward the final goal."),
            "requirements": _coerce_str_list(raw.get("requirements")),
            "success_criteria": _coerce_str_list(raw.get("success_criteria")),
        }
    return {
        "goal": _coerce_str(raw, "Continue progress toward the final goal."),
        "requirements": [],
        "success_criteria": [],
    }


def _normalize_task_assignment(raw: Any) -> dict[str, Any]:
    item = raw if isinstance(raw, dict) else {"description": raw}
    task_id = _coerce_str(item.get("id"), "T")
    return {
        "id": task_id,
        "description": _coerce_str(item.get("description") or item.get("goal"), "Complete the assigned task."),
        "assigned_agent": _coerce_str(item.get("assigned_agent"), "SECFilingResearchAgent"),
        "candidate_agents": _coerce_str_list(item.get("candidate_agents")),
        "dependencies": _coerce_str_list(item.get("dependencies")),
        "input": item.get("input") if isinstance(item.get("input"), dict) else {},
        "expected_output": _normalize_expected_output(item.get("expected_output")),
        "success_criteria": _coerce_str_list(item.get("success_criteria")),
        "estimated_cost": _clamp_int(item.get("estimated_cost", 1), 1, 100),
        "allowed_tools": _coerce_str_list(item.get("allowed_tools")),
    }


def _normalize_perspective_assignment(raw: Any) -> dict[str, Any]:
    item = raw if isinstance(raw, dict) else {"instruction": raw}
    return {
        "id": _coerce_str(item.get("id"), "p"),
        "perspective": _coerce_str(item.get("perspective"), "general_perspective"),
        "assigned_agent": _coerce_str(item.get("assigned_agent"), "SECFilingResearchAgent"),
        "candidate_agents": _coerce_str_list(item.get("candidate_agents")),
        "instruction": _coerce_str(item.get("instruction") or item.get("description"), "Solve from this perspective."),
        "input": item.get("input") if isinstance(item.get("input"), dict) else {},
        "expected_output": _normalize_expected_output(item.get("expected_output")),
        "success_criteria": _coerce_str_list(item.get("success_criteria")),
        "estimated_cost": _clamp_int(item.get("estimated_cost", 1), 1, 100),
        "allowed_tools": _coerce_str_list(item.get("allowed_tools")),
    }


def _normalize_expected_output(raw: Any) -> dict[str, Any]:
    if isinstance(raw, ExpectedOutput):
        return raw.model_dump(mode="json")
    if isinstance(raw, dict):
        return {
            "type": _coerce_str(raw.get("type"), "text"),
            "fields": _coerce_str_list(raw.get("fields")),
        }
    if isinstance(raw, str):
        return {"type": raw, "fields": []}
    return {"type": "text", "fields": []}


def _normalize_plan_rubric_check(raw: Any, *, default_passed: bool) -> dict[str, Any]:
    if isinstance(raw, PlanRubricCheck):
        return raw.model_dump(mode="json")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "passed": _coerce_bool(raw.get("passed", default_passed)),
        "score": _clamp_float(raw.get("score", 0.5), 0.0, 1.0),
        "issues": _coerce_str_list(raw.get("issues")),
        "refinements": _coerce_str_list(raw.get("refinements")),
    }


def _normalize_evidence_item(raw: Any) -> dict[str, Any]:
    if isinstance(raw, EvidenceItem):
        return raw.model_dump(mode="json")
    if not isinstance(raw, dict):
        raw = {"claim": raw}
    return {
        "claim": _coerce_str(raw.get("claim") or raw.get("text"), "Unspecified claim."),
        "value": None if raw.get("value") is None else _coerce_str(raw.get("value"), ""),
        "source_name": None if raw.get("source_name") is None else _coerce_str(raw.get("source_name"), ""),
        "source_url": None if raw.get("source_url") is None else _coerce_str(raw.get("source_url"), ""),
        "quote_or_location": None
        if raw.get("quote_or_location") is None
        else _coerce_str(raw.get("quote_or_location"), ""),
        "confidence": _clamp_float(raw.get("confidence", 0.5), 0.0, 1.0),
    }


def _normalize_task_assessment(raw: Any) -> dict[str, Any]:
    item = raw if isinstance(raw, dict) else {}
    return {
        "task_id": _coerce_str(item.get("task_id") or item.get("id"), "unknown"),
        "assigned_agent": _coerce_str(item.get("assigned_agent"), "unknown"),
        "completed": _coerce_bool(item.get("completed", False)),
        "score": _clamp_float(item.get("score", 0.5), 0.0, 1.0),
        "satisfied_criteria": _coerce_str_list(item.get("satisfied_criteria")),
        "missing_criteria": _coerce_str_list(item.get("missing_criteria")),
        "evidence_alignment": _coerce_str(item.get("evidence_alignment"), ""),
        "contradictions": _coerce_str_list(item.get("contradictions")),
        "confidence": _clamp_float(item.get("confidence", 0.5), 0.0, 1.0),
    }


def _normalize_capability_update(raw: dict[str, Any]) -> dict[str, Any]:
    outcome = str(raw.get("outcome", "mixed")).strip().lower().replace("_", " ")
    if outcome in {"success", "successful", "passed", "pass"}:
        normalized_outcome = "success"
    elif outcome in {"failure", "failed", "fail", "unsuccessful"}:
        normalized_outcome = "failure"
    else:
        normalized_outcome = "mixed"
    return {
        "agent_name": _coerce_str(raw.get("agent_name"), "unknown"),
        "task_type": _coerce_str(raw.get("task_type"), "unknown"),
        "outcome": normalized_outcome,
        "evidence": _coerce_str(raw.get("evidence"), ""),
        "score_delta": _clamp_float(raw.get("score_delta", 0.0), -0.2, 0.2),
    }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "pass", "passed", "success"}
    return False


def _coerce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return json_dumps_compact(value)


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_str_list(value: Any) -> list[str]:
    return [_coerce_str(item, "") for item in _coerce_list(value) if item is not None]


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = (low + high) / 2
    return min(high, max(low, parsed))


def _clamp_int(value: Any, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = low
    return min(high, max(low, parsed))


def json_dumps_compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
