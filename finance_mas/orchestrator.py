from __future__ import annotations

import json
from typing import Any, Callable

from model_library.base import LLM

from .llm_utils import query_json, query_text
from .prompts import (
    AGGREGATION_SYSTEM,
    DECIDER_SYSTEM,
    FINAL_ANSWER_SYSTEM,
    INSPECTOR_SYSTEM,
    PLANNER_SYSTEM,
    RUBRIC_SYSTEM,
    VALIDATOR_SYSTEM,
)
from .schemas import (
    AggregationResult,
    DeciderResult,
    EvidenceItem,
    ExpectedOutput,
    GoalState,
    InspectionResult,
    PerspectiveAssignment,
    Plan,
    RubricResult,
    SubagentOutput,
    TaskAssignment,
    TaskState,
    ValidationResult,
)
from .schema_utils import (
    parse_aggregation_result,
    parse_decider_result,
    parse_inspection_result,
    parse_plan,
    parse_rubric_result,
    parse_validation_result,
)

EventRecorder = Callable[..., None]


class Orchestrator:
    def __init__(self, llm: LLM, event_recorder: EventRecorder | None = None):
        self._llm = llm
        self._event_recorder = event_recorder

    async def decide(self, state: TaskState) -> DeciderResult:
        user = state.for_decider()
        self._record_llm_input("decider", state.iteration, DECIDER_SYSTEM, user)
        data = await query_json(self._llm, DECIDER_SYSTEM, user)
        self._record_llm_json("decider", state.iteration, data)
        decision = parse_decider_result(data)
        self._record_result("decider", state.iteration, decision.model_dump(mode="json"))
        if not decision.should_answer and decision.next_goal_state is None:
            decision.next_goal_state = GoalState(
                goal="Collect and verify the missing evidence needed for the answer.",
                requirements=["all key entities covered", "numerical result verified", "no unresolved contradictions"],
                success_criteria=["missing facts identified", "sources are traceable", "units and periods are explicit"],
            )
        return decision

    async def plan(self, state: TaskState) -> Plan:
        user = (
            f"{state.for_planner()}\n\n"
            "Plan for the current stage goal_state, not for the entire final answer unless the current state is ready."
        )
        self._record_llm_input("planner", state.iteration, PLANNER_SYSTEM, user)
        data = await query_json(self._llm, PLANNER_SYSTEM, user)
        self._record_llm_json("planner", state.iteration, data)
        plan = parse_plan(data)
        plan = self._with_plan_defaults(plan)
        self._record_result("planner", state.iteration, plan.model_dump(mode="json"))
        return plan

    async def inspect(self, state: TaskState, plan: Plan) -> InspectionResult:
        user = (
            f"Current state:\n{state.for_planner()}\n\n"
            f"Proposed plan:\n{plan.model_dump_json(indent=2)}"
        )
        self._record_llm_input("inspector", state.iteration, INSPECTOR_SYSTEM, user)
        data = await query_json(self._llm, INSPECTOR_SYSTEM, user)
        self._record_llm_json("inspector", state.iteration, data)
        inspection = parse_inspection_result(data)
        if inspection.refined_plan is not None:
            inspection.refined_plan = self._with_plan_defaults(inspection.refined_plan)
        self._record_result("inspector", state.iteration, inspection.model_dump(mode="json"))
        return inspection

    def _with_plan_defaults(self, plan: Plan) -> Plan:
        if plan.mode == "parallel_subtasks" and not plan.tasks:
            plan.mode = "perspective_debate"
        if plan.mode == "perspective_debate" and not plan.perspectives:
            plan.mode = "parallel_subtasks"
        if plan.mode == "parallel_subtasks" and not plan.tasks:
            plan.tasks = [
                TaskAssignment(
                    id="T1",
                    description="Research the authoritative SEC filing evidence, filing period, units, and relevant disclosed figures.",
                    assigned_agent="SECFilingResearchAgent",
                    candidate_agents=["SECFilingResearchAgent"],
                    input={"query": plan.goal_state.goal},
                    expected_output=ExpectedOutput(
                        type="sec_evidence",
                        fields=["company", "cik", "filing_type", "period", "values", "units", "source_url"],
                    ),
                    success_criteria=[
                        "authoritative SEC filing identified",
                        "period and units specified",
                        "relevant disclosed figures extracted with source location",
                    ],
                    allowed_tools=["edgar_search", "parse_html_page", "retrieve_information", "web_search"],
                ),
                TaskAssignment(
                    id="T2",
                    description="Check whether regulatory actions, news, M&A events, or market messages are needed to interpret the question.",
                    assigned_agent="RegulatoryNewsAgent",
                    candidate_agents=["RegulatoryNewsAgent"],
                    input={"query": plan.goal_state.goal},
                    expected_output=ExpectedOutput(
                        type="event_context",
                        fields=["event_or_message", "date", "source", "relevance", "risk_or_catalyst"],
                    ),
                    success_criteria=[
                        "relevant external events identified or ruled out",
                        "source and date provided when events matter",
                        "relationship to the finance question explained",
                    ],
                    allowed_tools=["web_search", "parse_html_page", "retrieve_information"],
                ),
                TaskAssignment(
                    id="T3",
                    description="Assess operational implications and sanity-check how the evidence affects revenue, costs, risks, or market exposure.",
                    assigned_agent="OperationalImpactAgent",
                    candidate_agents=["OperationalImpactAgent"],
                    dependencies=["T1"],
                    input={"query": plan.goal_state.goal},
                    expected_output=ExpectedOutput(
                        type="operational_assessment",
                        fields=["revenue_impact", "cost_impact", "business_risk", "market_impact", "confidence"],
                    ),
                    success_criteria=[
                        "operational effect supported by evidence",
                        "numeric implications are sanity-checked",
                        "remaining business risks or uncertainties stated",
                    ],
                    allowed_tools=["retrieve_information", "web_search", "parse_html_page"],
                ),
                TaskAssignment(
                    id="T4",
                    description="Verify all numeric values, formulas, units, fiscal periods, signs, denominators, and rounding used for the stage goal.",
                    assigned_agent="NumericalVerificationAgent",
                    candidate_agents=["NumericalVerificationAgent"],
                    dependencies=["T1"],
                    input={"query": plan.goal_state.goal},
                    expected_output=ExpectedOutput(
                        type="numeric_verification",
                        fields=["verified_values", "formula", "unit_check", "period_check", "rounding_check", "issues"],
                    ),
                    success_criteria=[
                        "numeric values traced to evidence",
                        "formula and arithmetic checked",
                        "units, periods, and rounding verified",
                    ],
                    allowed_tools=["retrieve_information", "parse_html_page"],
                ),
                TaskAssignment(
                    id="T5",
                    description="Audit evidence quality, source hierarchy, and source-to-claim alignment for the facts collected so far.",
                    assigned_agent="SourceGroundingAgent",
                    candidate_agents=["SourceGroundingAgent"],
                    dependencies=["T1", "T2"],
                    input={"query": plan.goal_state.goal},
                    expected_output=ExpectedOutput(
                        type="source_grounding_audit",
                        fields=["claim", "best_source", "source_rank", "support_status", "conflicts"],
                    ),
                    success_criteria=[
                        "primary sources preferred where available",
                        "each key claim has traceable support",
                        "source conflicts identified or ruled out",
                    ],
                    allowed_tools=["retrieve_information", "parse_html_page", "edgar_search", "web_search"],
                ),
                TaskAssignment(
                    id="T6",
                    description="Perform adversarial contradiction review against the emerging answer and identify alternative interpretations or invalidating evidence.",
                    assigned_agent="ContradictionAgent",
                    candidate_agents=["ContradictionAgent"],
                    dependencies=["T1", "T3", "T4", "T5"],
                    input={"query": plan.goal_state.goal},
                    expected_output=ExpectedOutput(
                        type="contradiction_review",
                        fields=["contradictions", "alternative_interpretations", "wrong_period_risks", "wrong_entity_risks", "resolution"],
                    ),
                    success_criteria=[
                        "alternative interpretations tested",
                        "period/entity/unit traps checked",
                        "unresolved contradictions stated",
                    ],
                    allowed_tools=["retrieve_information", "edgar_search", "web_search", "parse_html_page"],
                ),
            ]
        if plan.mode == "perspective_debate" and not plan.perspectives:
            plan.perspectives = [
                PerspectiveAssignment(
                    id="p1",
                    perspective="sec_filing_first",
                    assigned_agent="SECFilingResearchAgent",
                    candidate_agents=["SECFilingResearchAgent"],
                    instruction=(
                        "Solve the stage goal by prioritizing SEC/EDGAR filings, financial statements, exhibits, "
                        "filing periods, units, and primary-source evidence."
                    ),
                    input={"query": plan.goal_state.goal},
                    success_criteria=["SEC evidence identified", "period and units verified", "source traceable"],
                    allowed_tools=["edgar_search", "parse_html_page", "retrieve_information", "web_search"],
                ),
                PerspectiveAssignment(
                    id="p2",
                    perspective="regulatory_news_context",
                    assigned_agent="RegulatoryNewsAgent",
                    candidate_agents=["RegulatoryNewsAgent"],
                    instruction=(
                        "Solve the stage goal by checking whether regulatory developments, news, M&A, litigation, "
                        "product announcements, or market messages change interpretation of the evidence."
                    ),
                    input={"query": plan.goal_state.goal},
                    success_criteria=["relevant event context identified or ruled out", "sources dated", "interpretive impact explained"],
                    allowed_tools=["web_search", "retrieve_information"],
                ),
                PerspectiveAssignment(
                    id="p3",
                    perspective="operational_impact",
                    assigned_agent="OperationalImpactAgent",
                    candidate_agents=["OperationalImpactAgent"],
                    instruction=(
                        "Solve the stage goal by assessing revenue, cost, business risk, segment, market, and "
                        "competitive implications, and by sanity-checking whether the numeric evidence is operationally plausible."
                    ),
                    input={"query": plan.goal_state.goal},
                    success_criteria=["operational implications assessed", "numeric implications sanity-checked", "uncertainties stated"],
                    allowed_tools=["retrieve_information", "web_search", "parse_html_page"],
                ),
                PerspectiveAssignment(
                    id="p4",
                    perspective="numeric_verification",
                    assigned_agent="NumericalVerificationAgent",
                    candidate_agents=["NumericalVerificationAgent"],
                    instruction=(
                        "Solve the stage goal by auditing the numeric chain: values, formulas, denominators, units, "
                        "fiscal periods, signs, and rounding. Identify any arithmetic or interpretation errors."
                    ),
                    input={"query": plan.goal_state.goal},
                    success_criteria=["numeric chain audited", "units and periods checked", "rounding verified"],
                    allowed_tools=["retrieve_information", "parse_html_page"],
                ),
                PerspectiveAssignment(
                    id="p5",
                    perspective="source_grounding",
                    assigned_agent="SourceGroundingAgent",
                    candidate_agents=["SourceGroundingAgent"],
                    instruction=(
                        "Solve the stage goal by auditing whether each key claim is supported by the strongest "
                        "available source, with SEC filings and primary company sources ranked above secondary sources."
                    ),
                    input={"query": plan.goal_state.goal},
                    success_criteria=["claims tied to sources", "source hierarchy checked", "conflicts surfaced"],
                    allowed_tools=["retrieve_information", "parse_html_page", "edgar_search", "web_search"],
                ),
                PerspectiveAssignment(
                    id="p6",
                    perspective="contradiction_review",
                    assigned_agent="ContradictionAgent",
                    candidate_agents=["ContradictionAgent"],
                    instruction=(
                        "Try to falsify the emerging answer. Search for contradictions, alternative interpretations, "
                        "wrong-period or wrong-entity risks, restatements, unit traps, and evidence that would invalidate the answer."
                    ),
                    input={"query": plan.goal_state.goal},
                    success_criteria=["contradictions searched", "alternative interpretations tested", "remaining risks stated"],
                    allowed_tools=["retrieve_information", "edgar_search", "web_search", "parse_html_page"],
                ),
            ]
        if not plan.aggregation_rule:
            plan.aggregation_rule = "assess_each_task_then_deduplicate_align_evidence_and_resolve_conflicts"
        return plan

    async def aggregate(self, state: TaskState, plan: Plan, outputs: list[SubagentOutput]) -> AggregationResult:
        user = (
            f"Current state:\n{state.for_aggregation()}\n\n"
            f"Plan used for this execution:\n{plan.model_dump_json(indent=2)}\n\n"
            "New subagent outputs:\n"
            + json.dumps([output.model_dump(mode="json") for output in outputs], indent=2)
            + "\n\nDebate critiques:\n"
            + json.dumps([critique.model_dump(mode="json") for critique in state.critiques[-12:]], indent=2)
        )
        self._record_llm_input("aggregation", state.iteration, AGGREGATION_SYSTEM, user)
        data = await query_json(self._llm, AGGREGATION_SYSTEM, user)
        self._record_llm_json("aggregation", state.iteration, data)
        result = parse_aggregation_result(data)
        self._record_result("aggregation", state.iteration, result.model_dump(mode="json"))
        return result

    async def validate(self, state: TaskState, plan: Plan, aggregation: AggregationResult) -> ValidationResult:
        user = (
            f"Current stage goal_state:\n{plan.goal_state.model_dump_json(indent=2)}\n\n"
            f"Aggregation result:\n{aggregation.model_dump_json(indent=2)}\n\n"
            f"State:\n{state.for_validation()}\n\n"
            "Recent subagent outputs:\n"
            + json.dumps([item.model_dump(mode="json") for item in state.subagent_outputs[-8:]], indent=2)
        )
        self._record_llm_input("validation", state.iteration, VALIDATOR_SYSTEM, user)
        data = await query_json(self._llm, VALIDATOR_SYSTEM, user)
        self._record_llm_json("validation", state.iteration, data)
        result = parse_validation_result(data)
        self._record_result("validation", state.iteration, result.model_dump(mode="json"))
        return result

    async def final_answer(self, state: TaskState) -> str:
        user = (
            f"Question:\n{state.question}\n\n"
            f"State:\n{state.for_final_answer()}\n\n"
            "Accepted facts:\n"
            + json.dumps([item.model_dump(mode="json") for item in state.accepted_facts], indent=2)
        )
        self._record_llm_input("final_answer", state.iteration, FINAL_ANSWER_SYSTEM, user)
        answer = await query_text(self._llm, FINAL_ANSWER_SYSTEM, user)
        self._record_result("final_answer", state.iteration, {"answer": answer})
        return answer

    async def judge_final(self, state: TaskState, answer: str) -> RubricResult:
        user = (
            f"Question:\n{state.question}\n\n"
            f"Answer:\n{answer}\n\n"
            "Evidence state:\n"
            + json.dumps([item.model_dump(mode="json") for item in state.accepted_facts], indent=2)
        )
        self._record_llm_input("rubric_judge", state.iteration, RUBRIC_SYSTEM, user)
        data = await query_json(self._llm, RUBRIC_SYSTEM, user)
        self._record_llm_json("rubric_judge", state.iteration, data)
        result = parse_rubric_result(data)
        self._record_result("rubric_judge", state.iteration, result.model_dump(mode="json"))
        return result

    def _record_llm_input(self, component: str, iteration: int, system: str, user: str) -> None:
        if self._event_recorder is not None:
            self._event_recorder(
                "llm_input",
                component=component,
                iteration=iteration,
                payload={"system": system, "user": user},
            )

    def _record_llm_json(self, component: str, iteration: int, data: dict[str, Any]) -> None:
        if self._event_recorder is not None:
            self._event_recorder("llm_json_output", component=component, iteration=iteration, payload=data)

    def _record_result(self, component: str, iteration: int, result: dict[str, Any]) -> None:
        if self._event_recorder is not None:
            self._event_recorder("component_result", component=component, iteration=iteration, payload=result)
