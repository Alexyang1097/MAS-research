from __future__ import annotations

import asyncio
import logging
from typing import Any

from model_library.base import LLM
from pydantic import BaseModel, Field

from finance_agent.tools import (
    EDGARSearch,
    ParseHtmlPage,
    RetrieveInformation,
    TavilyWebSearch,
    Tool,
)

from .orchestrator import Orchestrator
from .schemas import CapabilityUpdate, Plan, TaskState
from .subagent_registry import default_capability_profiles
from .subagents import SubagentRunner, ToolExecutor


class MASParameters(BaseModel):
    max_iterations: int = 6
    max_rubric_failures: int = 2
    max_tool_rounds_per_subagent: int = 3
    max_tool_calls_per_round: int = 3
    debate_rounds_cap: int = 2
    max_plan_refinements: int = 1
    tools: list[str] = Field(
        default_factory=lambda: ["web_search", "edgar_search", "parse_html_page", "retrieve_information"]
    )


class MASRunResult(BaseModel):
    question: str
    final_answer: str
    success: bool
    iterations: int
    rubric_failures: int
    state: TaskState


class MASRunner:
    def __init__(
        self,
        llm: LLM,
        parameters: MASParameters | None = None,
        logger: logging.Logger | None = None,
    ):
        self._llm = llm
        self._parameters = parameters or MASParameters()
        self._logger = logger or logging.getLogger(__name__)
        self._tool_state: dict[str, Any] = {}
        self._tools = self._build_tools()
        self._tool_executor = ToolExecutor(self._tools, self._tool_state, self._logger)
        self._subagents = SubagentRunner(
            llm=llm,
            tool_executor=self._tool_executor,
            logger=self._logger,
            max_tool_rounds=self._parameters.max_tool_rounds_per_subagent,
            max_tool_calls_per_round=self._parameters.max_tool_calls_per_round,
        )
        self._orchestrator = Orchestrator(llm)

    async def run(self, question: str) -> MASRunResult:
        state = TaskState(question=question, capability_profiles=default_capability_profiles())
        final_answer = ""
        rubric_failures = 0

        for iteration in range(1, self._parameters.max_iterations + 1):
            state.iteration = iteration
            state.tool_state_keys = self._tool_executor.state_keys

            decision = await self._orchestrator.decide(state)
            state.decisions.append(decision)
            if decision.next_goal_state is not None:
                state.goal_state = decision.next_goal_state
            if decision.should_answer:
                final_answer = await self._orchestrator.final_answer(state)
                rubric = await self._orchestrator.judge_final(state, final_answer)
                state.rubric_results.append(rubric)
                if rubric.passed:
                    return MASRunResult(
                        question=question,
                        final_answer=final_answer,
                        success=True,
                        iterations=iteration,
                        rubric_failures=rubric_failures,
                        state=state,
                    )
                rubric_failures += 1
                state.candidate_answer = final_answer
                state.unresolved_issues = list(rubric.required_fixes)
                if rubric_failures >= self._parameters.max_rubric_failures:
                    break

            plan = await self._plan_with_inspection(state)
            state.goal_state = plan.goal_state
            state.plans.append(plan)

            outputs = await self._execute_plan(plan, state)
            state.subagent_outputs.extend(outputs)

            if plan.mode == "perspective_debate" and outputs:
                critiques = await self._run_debate_rounds(plan, state, outputs)
                state.critiques.extend(critiques)

            aggregation = await self._orchestrator.aggregate(state, plan, outputs)
            state.aggregations.append(aggregation)
            state.accepted_facts.extend(aggregation.accepted_facts)
            if aggregation.candidate_answer:
                state.candidate_answer = aggregation.candidate_answer
            state.unresolved_issues = aggregation.unresolved_issues
            state.tool_state_keys = self._tool_executor.state_keys

            validation = await self._orchestrator.validate(state, plan, aggregation)
            state.validations.append(validation)
            self._apply_capability_updates(state, validation.capability_updates)

            if validation.passed or plan.ready_for_final:
                final_answer = await self._orchestrator.final_answer(state)
                rubric = await self._orchestrator.judge_final(state, final_answer)
                state.rubric_results.append(rubric)
                if rubric.passed:
                    return MASRunResult(
                        question=question,
                        final_answer=final_answer,
                        success=True,
                        iterations=iteration,
                        rubric_failures=rubric_failures,
                        state=state,
                    )
                rubric_failures += 1
                state.candidate_answer = final_answer
                state.unresolved_issues = list(rubric.required_fixes)
                if rubric_failures >= self._parameters.max_rubric_failures:
                    break

        if not final_answer:
            final_answer = await self._orchestrator.final_answer(state)
        return MASRunResult(
            question=question,
            final_answer=final_answer,
            success=False,
            iterations=state.iteration,
            rubric_failures=rubric_failures,
            state=state,
        )

    async def _plan_with_inspection(self, state: TaskState) -> Plan:
        plan = await self._orchestrator.plan(state)
        for _refine_index in range(self._parameters.max_plan_refinements + 1):
            inspection = await self._orchestrator.inspect(state, plan)
            state.inspections.append(inspection)
            if inspection.passed or inspection.refined_plan is None:
                return plan
            plan = inspection.refined_plan
        return plan

    def _apply_capability_updates(self, state: TaskState, updates: list[CapabilityUpdate]) -> None:
        for update in updates:
            profile = state.capability_profiles.get(update.agent_name)
            if profile is None:
                continue

            if update.outcome == "success":
                profile.successes += 1
                entry = f"{update.task_type}: {update.evidence}"
                if entry not in profile.representative_success_tasks:
                    profile.representative_success_tasks.append(entry)
            elif update.outcome == "failure":
                profile.failures += 1
                entry = f"{update.task_type}: {update.evidence}"
                if entry not in profile.failure_cases:
                    profile.failure_cases.append(entry)
            else:
                entry = f"mixed {update.task_type}: {update.evidence}"
                if entry not in profile.representative_success_tasks:
                    profile.representative_success_tasks.append(entry)

            profile.score = min(1.0, max(0.0, profile.score + update.score_delta))

    async def _execute_plan(self, plan: Plan, state: TaskState):
        if plan.mode == "parallel_subtasks":
            return await self._execute_parallel_subtasks(plan, state)
        return await self._execute_perspectives(plan, state)

    async def _execute_parallel_subtasks(self, plan: Plan, state: TaskState):
        completed: dict[str, Any] = {}
        pending = list(plan.tasks)
        outputs = []

        while pending:
            ready = [
                assignment
                for assignment in pending
                if all(dependency in completed for dependency in assignment.dependencies)
            ]
            if not ready:
                ready = pending[:1]

            batch_outputs = await asyncio.gather(
                *[
                    self._subagents.run_assignment(
                        agent_name=assignment.assigned_agent,
                        assignment_id=assignment.id,
                        goal=assignment.description,
                        global_context=state.compact_context(),
                        allowed_tools=assignment.allowed_tools,
                        extra_instruction=(
                            f"Structured task input:\n{assignment.input}\n"
                            f"Candidate agents considered: {assignment.candidate_agents}\n"
                            f"Expected output: {assignment.expected_output.model_dump(mode='json')}\n"
                            f"Success criteria: {assignment.success_criteria}\n"
                            f"Estimated cost: {assignment.estimated_cost}"
                        ),
                    )
                    for assignment in ready
                ]
            )
            outputs.extend(batch_outputs)
            for output in batch_outputs:
                completed[output.assignment_id] = output
            ready_ids = {assignment.id for assignment in ready}
            pending = [assignment for assignment in pending if assignment.id not in ready_ids]

        return outputs

    async def _execute_perspectives(self, plan: Plan, state: TaskState):
        return await asyncio.gather(
            *[
                self._subagents.run_assignment(
                    agent_name=perspective.assigned_agent,
                    assignment_id=perspective.id,
                    goal=plan.goal_state.goal,
                    global_context=state.compact_context(),
                    allowed_tools=perspective.allowed_tools,
                    extra_instruction=(
                        f"Perspective: {perspective.perspective}\n"
                        f"Instruction: {perspective.instruction}\n"
                        f"Structured input: {perspective.input}\n"
                        f"Candidate agents considered: {perspective.candidate_agents}\n"
                        f"Expected output: {perspective.expected_output.model_dump(mode='json')}\n"
                        f"Success criteria: {perspective.success_criteria}\n"
                        "Solve the whole current goal from this perspective, not a small subpart."
                    ),
                )
                for perspective in plan.perspectives
            ]
        )

    async def _run_debate_rounds(self, plan: Plan, state: TaskState, outputs):
        rounds = min(plan.debate_rounds, self._parameters.debate_rounds_cap)
        critiques = []
        for _round_index in range(rounds):
            critiques.extend(
                await asyncio.gather(
                    *[
                        self._subagents.critique(
                            agent_name=output.agent_name,
                            global_context=state.compact_context(),
                            own_output=output,
                            all_outputs=outputs,
                        )
                        for output in outputs
                    ]
                )
            )
        return critiques

    def _build_tools(self) -> list[Tool]:
        available: dict[str, Tool] = {}
        if "web_search" in self._parameters.tools:
            available["web_search"] = TavilyWebSearch()
        if "edgar_search" in self._parameters.tools:
            available["edgar_search"] = EDGARSearch()
        if "parse_html_page" in self._parameters.tools:
            available["parse_html_page"] = ParseHtmlPage()
        if "retrieve_information" in self._parameters.tools:
            available["retrieve_information"] = RetrieveInformation(llm=self._llm)
        return list(available.values())
