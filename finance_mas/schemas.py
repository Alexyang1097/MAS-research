from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DecompositionMode = Literal["parallel_subtasks", "perspective_debate"]


class GoalState(BaseModel):
    goal: str
    requirements: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)

    def compact(self) -> str:
        return (
            f"Goal: {self.goal}\n"
            f"Requirements: {self.requirements}\n"
            f"Success criteria: {self.success_criteria}"
        )


class EvidenceItem(BaseModel):
    claim: str
    value: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    quote_or_location: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ToolCallRequest(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    purpose: str


class ExpectedOutput(BaseModel):
    type: str
    fields: list[str] = Field(default_factory=list)


class TaskAssignment(BaseModel):
    id: str
    description: str
    assigned_agent: str
    candidate_agents: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    input: dict[str, Any] = Field(default_factory=dict)
    expected_output: ExpectedOutput
    success_criteria: list[str] = Field(default_factory=list)
    estimated_cost: int = Field(default=1, ge=1)
    allowed_tools: list[str] = Field(default_factory=list)


class PerspectiveAssignment(BaseModel):
    id: str
    perspective: str
    assigned_agent: str
    candidate_agents: list[str] = Field(default_factory=list)
    instruction: str
    input: dict[str, Any] = Field(default_factory=dict)
    expected_output: ExpectedOutput = Field(
        default_factory=lambda: ExpectedOutput(type="reasoned_answer", fields=["answer", "evidence", "risks"])
    )
    success_criteria: list[str] = Field(default_factory=list)
    estimated_cost: int = Field(default=1, ge=1)
    allowed_tools: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    goal_state: GoalState
    decomposition_diagnosis: str
    mode: DecompositionMode
    tasks: list[TaskAssignment] = Field(default_factory=list)
    perspectives: list[PerspectiveAssignment] = Field(default_factory=list)
    debate_rounds: int = Field(default=1, ge=1, le=3)
    aggregation_rule: str = "assess_each_task_then_deduplicate_align_evidence_and_resolve_conflicts"
    ready_for_final: bool = False


class DeciderResult(BaseModel):
    should_answer: bool
    reason: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    answer_strategy: str | None = None
    next_goal_state: GoalState | None = None


class PlanRubricCheck(BaseModel):
    passed: bool
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    refinements: list[str] = Field(default_factory=list)


class InspectionResult(BaseModel):
    passed: bool
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    completeness: PlanRubricCheck
    solvability: PlanRubricCheck
    assignment_fit: PlanRubricCheck
    redundancy: PlanRubricCheck
    issues: list[str] = Field(default_factory=list)
    refinements: list[str] = Field(default_factory=list)
    refined_plan: Plan | None = None


class SubagentOutput(BaseModel):
    agent_name: str
    assignment_id: str
    goal: str
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    calculations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CritiqueOutput(BaseModel):
    agent_name: str = ""
    target_assignment_ids: list[str] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    recommended_revision: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    achieved: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    suggested_next_goal: str | None = None
    capability_updates: list["CapabilityUpdate"] = Field(default_factory=list)


class CapabilityProfile(BaseModel):
    agent_name: str
    description: str
    capability_tags: list[str] = Field(default_factory=list)
    representative_success_tasks: list[str] = Field(default_factory=list)
    failure_cases: list[str] = Field(default_factory=list)
    successes: int = 0
    failures: int = 0
    score: float = Field(default=0.5, ge=0.0, le=1.0)


class CapabilityUpdate(BaseModel):
    agent_name: str
    task_type: str
    outcome: Literal["success", "failure", "mixed"]
    evidence: str
    score_delta: float = Field(default=0.0, ge=-0.2, le=0.2)


class TaskAssessment(BaseModel):
    task_id: str
    assigned_agent: str
    completed: bool
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    satisfied_criteria: list[str] = Field(default_factory=list)
    missing_criteria: list[str] = Field(default_factory=list)
    evidence_alignment: str
    contradictions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AggregationResult(BaseModel):
    task_assessments: list[TaskAssessment] = Field(default_factory=list)
    accepted_facts: list[EvidenceItem] = Field(default_factory=list)
    duplicate_facts: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    evidence_alignment_summary: str = ""
    confidence_fusion: str = ""
    candidate_answer: str | None = None
    unresolved_issues: list[str] = Field(default_factory=list)


class RubricResult(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    numeric_accuracy: str
    logic_correctness: str
    evidence_quality: str
    answer_completeness: str
    required_fixes: list[str] = Field(default_factory=list)


class TaskState(BaseModel):
    question: str
    final_goal: GoalState = Field(default_factory=lambda: GoalState(
        goal="Produce the final answer to the finance benchmark question.",
        requirements=[
            "all key entities covered",
            "numerical result verified",
            "SEC or primary-source evidence used",
            "no unresolved contradictions",
            "sources dictionary included",
        ],
        success_criteria=[
            "answer directly addresses the query",
            "periods and units are correct",
            "calculations are arithmetically sound",
            "supporting evidence is traceable",
        ],
    ))
    goal_state: GoalState = Field(default_factory=lambda: GoalState(
        goal=(
        "Produce a final finance benchmark answer with correct SEC/primary-source evidence, exact periods, "
        "correct units, sound calculations, and a sources dictionary."
        ),
        requirements=["all key entities covered", "numerical result verified", "no unresolved contradictions"],
        success_criteria=["sufficient evidence", "correct metric definition", "correct arithmetic"],
    ))
    iteration: int = 0
    tool_state_keys: list[str] = Field(default_factory=list)
    capability_profiles: dict[str, CapabilityProfile] = Field(default_factory=dict)
    accepted_facts: list[EvidenceItem] = Field(default_factory=list)
    candidate_answer: str | None = None
    decisions: list[DeciderResult] = Field(default_factory=list)
    plans: list[Plan] = Field(default_factory=list)
    inspections: list[InspectionResult] = Field(default_factory=list)
    subagent_outputs: list[SubagentOutput] = Field(default_factory=list)
    critiques: list[CritiqueOutput] = Field(default_factory=list)
    aggregations: list[AggregationResult] = Field(default_factory=list)
    validations: list[ValidationResult] = Field(default_factory=list)
    rubric_results: list[RubricResult] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)

    def compact_context(self) -> str:
        parts = [
            f"Question: {self.question}",
            f"Final goal:\n{self.final_goal.compact()}",
            f"Current stage goal_state:\n{self.goal_state.compact()}",
            f"Iteration: {self.iteration}",
            f"Data storage keys: {', '.join(self.tool_state_keys) or '(none)'}",
        ]
        if self.capability_profiles:
            profiles = "\n".join(
                f"- {profile.agent_name}: score={profile.score:.2f}, successes={profile.successes}, failures={profile.failures}, "
                f"tags={profile.capability_tags}, success_examples={profile.representative_success_tasks[-3:]}, "
                f"failure_cases={profile.failure_cases[-3:]}"
                for profile in self.capability_profiles.values()
            )
            parts.append(f"Agent capability profiles:\n{profiles}")
        if self.decisions:
            decision = self.decisions[-1]
            parts.append(
                "Last decider result:\n"
                f"should_answer={decision.should_answer}, confidence={decision.confidence}, "
                f"next_goal_state={decision.next_goal_state.model_dump(mode='json') if decision.next_goal_state else None}, "
                f"reason={decision.reason}"
            )
        if self.aggregations:
            aggregation = self.aggregations[-1]
            parts.append(
                "Last aggregation:\n"
                f"task_assessments={[item.model_dump(mode='json') for item in aggregation.task_assessments]}, "
                f"conflicts={aggregation.conflicts}, unresolved={aggregation.unresolved_issues}"
            )
        if self.accepted_facts:
            facts = "\n".join(
                f"- {item.claim} | value={item.value or 'n/a'} | source={item.source_name or item.source_url or 'n/a'}"
                for item in self.accepted_facts[-12:]
            )
            parts.append(f"Accepted facts:\n{facts}")
        if self.candidate_answer:
            parts.append(f"Current candidate answer:\n{self.candidate_answer}")
        if self.unresolved_issues:
            parts.append("Unresolved issues:\n" + "\n".join(f"- {issue}" for issue in self.unresolved_issues[-10:]))
        if self.validations:
            last = self.validations[-1]
            parts.append(
                "Last validation/evaluation:\n"
                f"passed={last.passed}, score={last.score}, missing={last.missing}, concerns={last.concerns}, "
                f"suggested_next_goal={last.suggested_next_goal}"
            )
        if self.rubric_results:
            last_rubric = self.rubric_results[-1]
            parts.append(
                "Last rubric result:\n"
                f"passed={last_rubric.passed}, score={last_rubric.score}, required_fixes={last_rubric.required_fixes}"
            )
        return "\n\n".join(parts)
