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

    def is_equivalent_to(self, other: "GoalState") -> bool:
        return _normalize_goal_text(self.goal) == _normalize_goal_text(other.goal)


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


class FinalAttempt(BaseModel):
    iteration: int
    answer: str
    rubric: RubricResult
    trigger: str


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
    final_attempts: list[FinalAttempt] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)

    def for_decider(self) -> str:
        parts = [
            f"Question: {self.question}",
            f"Final goal:\n{self.final_goal.compact()}",
            f"Current stage goal_state:\n{self.goal_state.compact()}",
            f"Iteration: {self.iteration}",
        ]
        if self.candidate_answer:
            parts.append(f"Current candidate answer:\n{self.candidate_answer}")
        if self.accepted_facts:
            parts.append(f"Accepted facts summary:\n{self._facts_summary(limit=8)}")
        if self.unresolved_issues:
            parts.append("Unresolved issues:\n" + "\n".join(f"- {issue}" for issue in self.unresolved_issues[-10:]))
        if self.aggregations:
            aggregation = self.aggregations[-1]
            parts.append(
                "Last aggregation summary:\n"
                f"task_completion={self._task_assessment_summary(aggregation)}\n"
                f"conflicts={aggregation.conflicts}\n"
                f"confidence_fusion={aggregation.confidence_fusion}"
            )
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
        if self.final_attempts:
            parts.append(f"Final attempt pool: {self._final_attempts_summary()}")
        return "\n\n".join(parts)

    def for_planner(self) -> str:
        parts = [
            f"Question: {self.question}",
            f"Final goal:\n{self.final_goal.compact()}",
            f"Current stage goal_state:\n{self.goal_state.compact()}",
            f"Iteration: {self.iteration}",
            f"Data storage keys: {', '.join(self.tool_state_keys) or '(none)'}",
            f"Agent capability profiles:\n{self._capability_profiles_summary()}",
        ]
        if self.accepted_facts:
            parts.append(f"Accepted facts summary:\n{self._facts_summary(limit=8)}")
        if self.unresolved_issues:
            parts.append("Open issues to plan around:\n" + "\n".join(f"- {issue}" for issue in self.unresolved_issues[-8:]))
        if self.validations:
            last = self.validations[-1]
            parts.append(
                "Last validation feedback:\n"
                f"missing={last.missing}\nconcerns={last.concerns}\nsuggested_next_goal={last.suggested_next_goal}"
            )
        return "\n\n".join(parts)

    def for_subagent(self) -> str:
        parts = [
            f"Question: {self.question}",
            f"Current stage goal_state:\n{self.goal_state.compact()}",
            f"Data storage keys: {', '.join(self.tool_state_keys) or '(none)'}",
        ]
        if self.accepted_facts:
            parts.append(f"Accepted facts relevant so far:\n{self._facts_summary(limit=6)}")
        if self.unresolved_issues:
            parts.append("Unresolved issues:\n" + "\n".join(f"- {issue}" for issue in self.unresolved_issues[-6:]))
        if self.candidate_answer:
            parts.append(f"Current candidate answer, if relevant:\n{self.candidate_answer}")
        return "\n\n".join(parts)

    def for_aggregation(self) -> str:
        parts = [
            f"Question: {self.question}",
            f"Current stage goal_state:\n{self.goal_state.compact()}",
            f"Aggregation rule should update state while preserving conflicts and unsupported claims.",
        ]
        if self.accepted_facts:
            parts.append(f"Previously accepted facts:\n{self._facts_summary(limit=10)}")
        if self.unresolved_issues:
            parts.append("Prior unresolved issues:\n" + "\n".join(f"- {issue}" for issue in self.unresolved_issues[-8:]))
        return "\n\n".join(parts)

    def for_validation(self) -> str:
        parts = [
            f"Question: {self.question}",
            f"Final goal:\n{self.final_goal.compact()}",
            f"Current stage goal_state:\n{self.goal_state.compact()}",
            f"Accepted facts summary:\n{self._facts_summary(limit=10) if self.accepted_facts else '(none)'}",
        ]
        if self.candidate_answer:
            parts.append(f"Current candidate answer:\n{self.candidate_answer}")
        if self.unresolved_issues:
            parts.append("Unresolved issues:\n" + "\n".join(f"- {issue}" for issue in self.unresolved_issues[-10:]))
        return "\n\n".join(parts)

    def for_final_answer(self) -> str:
        parts = [
            f"Question: {self.question}",
            f"Final goal:\n{self.final_goal.compact()}",
        ]
        if self.candidate_answer:
            parts.append(f"Candidate answer to revise/finalize:\n{self.candidate_answer}")
        if self.accepted_facts:
            parts.append(f"Accepted evidence facts:\n{self._facts_summary(limit=20, include_location=True)}")
        if self.unresolved_issues:
            parts.append("Known unresolved issues or required fixes:\n" + "\n".join(f"- {issue}" for issue in self.unresolved_issues[-10:]))
        if self.rubric_results:
            last_rubric = self.rubric_results[-1]
            parts.append(
                "Last rubric feedback:\n"
                f"numeric_accuracy={last_rubric.numeric_accuracy}\n"
                f"logic_correctness={last_rubric.logic_correctness}\n"
                f"evidence_quality={last_rubric.evidence_quality}\n"
                f"required_fixes={last_rubric.required_fixes}"
            )
        if self.final_attempts:
            parts.append(f"Previous final attempts:\n{self._final_attempts_summary()}")
        return "\n\n".join(parts)

    def compact_context(self) -> str:
        return self.for_planner()

    def _facts_summary(self, limit: int, include_location: bool = False) -> str:
        lines = []
        for item in self.accepted_facts[-limit:]:
            source = item.source_name or item.source_url or "n/a"
            line = f"- {item.claim} | value={item.value or 'n/a'} | source={source} | confidence={item.confidence:.2f}"
            if include_location and item.quote_or_location:
                line += f" | location={item.quote_or_location}"
            lines.append(line)
        return "\n".join(lines)

    def _capability_profiles_summary(self) -> str:
        if not self.capability_profiles:
            return "(none)"
        return "\n".join(
            f"- {profile.agent_name}: description={profile.description}, score={profile.score:.2f}, "
            f"tags={profile.capability_tags}, "
            f"success_examples={profile.representative_success_tasks[-2:]}, failures={profile.failure_cases[-2:]}"
            for profile in self.capability_profiles.values()
        )

    def _task_assessment_summary(self, aggregation: AggregationResult) -> list[dict[str, object]]:
        return [
            {
                "task_id": item.task_id,
                "agent": item.assigned_agent,
                "completed": item.completed,
                "score": item.score,
                "missing": item.missing_criteria,
                "contradictions": item.contradictions,
            }
            for item in aggregation.task_assessments
        ]

    def _final_attempts_summary(self) -> list[dict[str, object]]:
        return [
            {
                "iteration": attempt.iteration,
                "score": attempt.rubric.score,
                "passed": attempt.rubric.passed,
                "trigger": attempt.trigger,
                "required_fixes": attempt.rubric.required_fixes,
            }
            for attempt in self.final_attempts[-5:]
        ]


def _normalize_goal_text(text: str) -> str:
    return " ".join(text.lower().strip().split())
