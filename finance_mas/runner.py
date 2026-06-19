from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
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
from .schemas import CapabilityUpdate, FinalAttempt, Plan, TaskState
from .subagent_registry import default_capability_profiles
from .subagents import SubagentRunner, ToolExecutor


class MASParameters(BaseModel):
    max_iterations: int = 30
    max_final_goal_replans: int = 3
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
        event_path: Path | str | None = None,
    ):
        self._llm = llm
        self._parameters = parameters or MASParameters()
        base_logger = logger or logging.getLogger(__name__)
        self._event_path = Path(event_path) if event_path is not None else None
        self._detail_dir = self._event_path.with_name(f"{self._event_path.stem}_details") if self._event_path else None
        self._event_seq = 0
        self._event_lock = threading.Lock()
        self._logger = self._build_run_logger(base_logger)
        self._configure_file_logger()
        self._tool_state: dict[str, Any] = {}
        self._tools = self._build_tools()
        self._tool_executor = ToolExecutor(self._tools, self._tool_state, self._logger)
        self._subagents = SubagentRunner(
            llm=llm,
            tool_executor=self._tool_executor,
            logger=self._logger,
            max_tool_rounds=self._parameters.max_tool_rounds_per_subagent,
            max_tool_calls_per_round=self._parameters.max_tool_calls_per_round,
            event_recorder=self._record_event,
        )
        self._orchestrator = Orchestrator(llm, event_recorder=self._record_event)

    async def run(self, question: str) -> MASRunResult:
        state = TaskState(question=question, capability_profiles=default_capability_profiles())
        final_answer = ""
        rubric_failures = 0
        final_goal_replans = 0
        self._record_event(
            "run_started",
            component="runner",
            payload={
                "question": question,
                "parameters": self._parameters.model_dump(mode="json"),
                "tools": [tool.name for tool in self._tools],
            },
        )

        for iteration in range(1, self._parameters.max_iterations + 1):
            state.iteration = iteration
            state.tool_state_keys = self._tool_executor.state_keys
            self._record_event(
                "iteration_started",
                component="runner",
                iteration=iteration,
                payload={
                    "goal_state": state.goal_state.model_dump(mode="json"),
                    "tool_state_keys": state.tool_state_keys,
                    "candidate_answer": state.candidate_answer,
                    "unresolved_issues": state.unresolved_issues,
                },
            )

            decision = await self._orchestrator.decide(state)
            state.decisions.append(decision)
            next_goal_is_final = (
                decision.next_goal_state is not None
                and decision.next_goal_state.is_equivalent_to(state.final_goal)
            )

            if decision.next_goal_state is not None and not next_goal_is_final:
                state.goal_state = decision.next_goal_state
                final_goal_replans = 0

            if decision.should_answer or next_goal_is_final:
                attempt_result = await self._attempt_final_answer(
                    state=state,
                    iteration=iteration,
                    trigger="decider_should_answer" if decision.should_answer else "decider_next_goal_is_final_goal",
                )
                final_answer = attempt_result.final_answer
                rubric_failures = attempt_result.rubric_failures
                if attempt_result.success:
                    return attempt_result

                final_goal_replans += 1
                if (
                    rubric_failures >= self._parameters.max_rubric_failures
                    or final_goal_replans >= self._parameters.max_final_goal_replans
                ):
                    best_attempt = self._best_final_attempt(state)
                    if best_attempt is not None:
                        return MASRunResult(
                            question=question,
                            final_answer=best_attempt.answer,
                            success=False,
                            iterations=iteration,
                            rubric_failures=rubric_failures,
                            state=state,
                        )
                    break

                state.goal_state = state.final_goal

            plan = await self._plan_with_inspection(state)
            state.goal_state = plan.goal_state
            state.plans.append(plan)
            self._record_event(
                "collaboration_selected",
                component="runner",
                iteration=iteration,
                payload={
                    "mode": plan.mode,
                    "tasks": [task.model_dump(mode="json") for task in plan.tasks],
                    "perspectives": [perspective.model_dump(mode="json") for perspective in plan.perspectives],
                    "aggregation_rule": plan.aggregation_rule,
                    "ready_for_final": plan.ready_for_final,
                },
            )

            outputs = await self._execute_plan(plan, state)
            state.subagent_outputs.extend(outputs)
            self._record_event(
                "plan_execution_finished",
                component="runner",
                iteration=iteration,
                payload={"outputs": [output.model_dump(mode="json") for output in outputs]},
            )

            if plan.mode == "perspective_debate" and outputs:
                critiques = await self._run_debate_rounds(plan, state, outputs)
                state.critiques.extend(critiques)
                self._record_event(
                    "debate_finished",
                    component="runner",
                    iteration=iteration,
                    payload={"critiques": [critique.model_dump(mode="json") for critique in critiques]},
                )

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
                attempt_result = await self._attempt_final_answer(
                    state=state,
                    iteration=iteration,
                    trigger="validation_passed_or_plan_ready_for_final",
                )
                final_answer = attempt_result.final_answer
                rubric_failures = attempt_result.rubric_failures
                if attempt_result.success:
                    return attempt_result
                final_goal_replans += 1
                if (
                    rubric_failures >= self._parameters.max_rubric_failures
                    or final_goal_replans >= self._parameters.max_final_goal_replans
                ):
                    best_attempt = self._best_final_attempt(state)
                    if best_attempt is not None:
                        return MASRunResult(
                            question=question,
                            final_answer=best_attempt.answer,
                            success=False,
                            iterations=iteration,
                            rubric_failures=rubric_failures,
                            state=state,
                        )
                    break

        best_attempt = self._best_final_attempt(state)
        if best_attempt is not None:
            final_answer = best_attempt.answer
        elif not final_answer:
            final_answer = await self._orchestrator.final_answer(state)
        result = MASRunResult(
            question=question,
            final_answer=final_answer,
            success=False,
            iterations=state.iteration,
            rubric_failures=rubric_failures,
            state=state,
        )
        self._record_event("run_finished", component="runner", payload=result.model_dump(mode="json"))
        return result

    async def _attempt_final_answer(self, state: TaskState, iteration: int, trigger: str) -> MASRunResult:
        self._record_event("final_attempt_started", component="runner", iteration=iteration, payload={"trigger": trigger})
        final_answer = await self._orchestrator.final_answer(state)
        rubric = await self._orchestrator.judge_final(state, final_answer)
        state.rubric_results.append(rubric)
        state.final_attempts.append(
            FinalAttempt(
                iteration=iteration,
                answer=final_answer,
                rubric=rubric,
                trigger=trigger,
            )
        )
        if rubric.passed:
            result = MASRunResult(
                question=state.question,
                final_answer=final_answer,
                success=True,
                iterations=iteration,
                rubric_failures=sum(1 for attempt in state.final_attempts if not attempt.rubric.passed),
                state=state,
            )
            self._record_event("run_finished", component="runner", iteration=iteration, payload=result.model_dump(mode="json"))
            return result
        state.candidate_answer = final_answer
        state.unresolved_issues = list(rubric.required_fixes)
        result = MASRunResult(
            question=state.question,
            final_answer=final_answer,
            success=False,
            iterations=iteration,
            rubric_failures=sum(1 for attempt in state.final_attempts if not attempt.rubric.passed),
            state=state,
        )
        self._record_event("final_attempt_finished", component="runner", iteration=iteration, payload=result.model_dump(mode="json"))
        return result

    def _best_final_attempt(self, state: TaskState) -> FinalAttempt | None:
        if not state.final_attempts:
            return None
        return max(state.final_attempts, key=lambda attempt: attempt.rubric.score)

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
            self._record_event(
                "subtask_batch_started",
                component="runner",
                iteration=state.iteration,
                payload={
                    "ready_task_ids": [assignment.id for assignment in ready],
                    "pending_task_ids": [assignment.id for assignment in pending],
                    "completed_task_ids": list(completed.keys()),
                    "dependency_inputs": {
                        assignment.id: {
                            "declared_dependencies": assignment.dependencies,
                            "available_dependencies": [
                                dependency for dependency in assignment.dependencies if dependency in completed
                            ],
                            "missing_dependencies": [
                                dependency for dependency in assignment.dependencies if dependency not in completed
                            ],
                        }
                        for assignment in ready
                    },
                },
            )

            batch_outputs = await asyncio.gather(
                *[
                    self._subagents.run_assignment(
                        agent_name=assignment.assigned_agent,
                        assignment_id=assignment.id,
                        goal=assignment.description,
                        global_context=state.for_subagent(),
                        allowed_tools=assignment.allowed_tools,
                        extra_instruction=self._subtask_extra_instruction(assignment, completed),
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

    def _subtask_extra_instruction(self, assignment: Any, completed: dict[str, Any]) -> str:
        parts = [
            f"Structured task input:\n{assignment.input}",
            f"Candidate agents considered: {assignment.candidate_agents}",
            f"Expected output: {assignment.expected_output.model_dump(mode='json')}",
            f"Success criteria: {assignment.success_criteria}",
            f"Estimated cost: {assignment.estimated_cost}",
        ]
        if assignment.dependencies:
            available_outputs = {
                dependency: self._compact_dependency_output(completed[dependency])
                for dependency in assignment.dependencies
                if dependency in completed
            }
            missing_dependencies = [
                dependency for dependency in assignment.dependencies if dependency not in completed
            ]
            parts.append(
                "Dependency outputs available to this task:\n"
                + json.dumps(available_outputs, indent=2, ensure_ascii=False, default=str)
            )
            if missing_dependencies:
                parts.append(
                    "Declared dependencies not available yet:\n"
                    + json.dumps(missing_dependencies, ensure_ascii=False)
                    + "\nProceed only if the runner scheduled you as a cycle-breaker; explicitly note this limitation."
                )
            else:
                parts.append(
                    "Use the dependency outputs above as direct upstream evidence. Cross-check them, preserve useful "
                    "source details, and explicitly mention any conflict or uncertainty you find."
                )
        return "\n\n".join(parts)

    def _compact_dependency_output(self, output: Any) -> dict[str, Any]:
        data = output.model_dump(mode="json") if hasattr(output, "model_dump") else dict(output)
        compact = {
            "agent_name": data.get("agent_name"),
            "assignment_id": data.get("assignment_id"),
            "goal": data.get("goal"),
            "answer": self._truncate_for_prompt(data.get("answer", ""), limit=12000),
            "evidence": data.get("evidence", []),
            "calculations": data.get("calculations", []),
            "assumptions": data.get("assumptions", []),
            "open_questions": data.get("open_questions", []),
            "failure_modes": data.get("failure_modes", []),
            "confidence": data.get("confidence"),
        }
        return compact

    def _truncate_for_prompt(self, value: Any, *, limit: int) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"

    async def _execute_perspectives(self, plan: Plan, state: TaskState):
        self._record_event(
            "perspective_batch_started",
            component="runner",
            iteration=state.iteration,
            payload={"perspective_ids": [perspective.id for perspective in plan.perspectives]},
        )
        return await asyncio.gather(
            *[
                self._subagents.run_assignment(
                    agent_name=perspective.assigned_agent,
                    assignment_id=perspective.id,
                    goal=plan.goal_state.goal,
                    global_context=state.for_subagent(),
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
                            global_context=state.for_aggregation(),
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

    def _configure_file_logger(self) -> None:
        if self._event_path is None:
            return
        self._event_path.parent.mkdir(parents=True, exist_ok=True)
        if self._detail_dir is not None:
            self._detail_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._agent_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_path.resolve())
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved:
                return
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def _build_run_logger(self, base_logger: logging.Logger) -> logging.Logger:
        if self._event_path is None:
            return base_logger
        return logging.getLogger(f"{base_logger.name}.mas_run.{self._event_path.stem}")

    def _record_event(
        self,
        event_type: str,
        *,
        component: str,
        iteration: int | None = None,
        assignment_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._event_path is None:
            return
        payload = payload or {}
        with self._event_lock:
            self._event_seq += 1
            seq = self._event_seq
            timestamp = datetime.now(timezone.utc).isoformat()
            detail_path = self._write_event_detail(
                seq=seq,
                timestamp=timestamp,
                event_type=event_type,
                component=component,
                iteration=iteration,
                assignment_id=assignment_id,
                payload=payload,
            )
            preview = self._payload_preview(payload)
            event = {
                "seq": seq,
                "timestamp": timestamp,
                "event_type": event_type,
                "component": component,
                "iteration": iteration,
                "assignment_id": assignment_id,
                "detail_path": str(detail_path),
                "payload_preview": preview,
            }
            with self._event_path.open("a") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._logger.info(self._format_event_log(event))

    def _write_event_detail(
        self,
        *,
        seq: int,
        timestamp: str,
        event_type: str,
        component: str,
        iteration: int | None,
        assignment_id: str | None,
        payload: dict[str, Any],
    ) -> Path:
        if self._detail_dir is None:
            raise RuntimeError("MAS detail directory is not configured.")
        safe_assignment = f"_{self._safe_name(assignment_id)}" if assignment_id else ""
        detail_path = self._detail_dir / f"{seq:06d}_{self._safe_name(component)}_{self._safe_name(event_type)}{safe_assignment}.json"
        with detail_path.open("w") as f:
            json.dump(
                {
                    "seq": seq,
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "component": component,
                    "iteration": iteration,
                    "assignment_id": assignment_id,
                    "payload": payload,
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        return detail_path

    def _format_event_log(self, event: dict[str, Any]) -> str:
        preview = event["payload_preview"]
        parts = [
            f"{self._event_action(event['event_type'])} {event['component']}",
            f"event={event['event_type']}",
        ]
        if event["iteration"] is not None:
            parts.append(f"iter={event['iteration']}")
        if event["assignment_id"]:
            parts.append(f"assignment={event['assignment_id']}")
        for key in (
            "agent_name",
            "round",
            "mode",
            "tool_name",
            "ready_task_ids",
            "task_ids",
            "perspective_ids",
            "trigger",
            "error",
            "skipped",
            "should_answer",
            "confidence",
            "passed",
            "score",
            "completed",
            "ready_for_final",
        ):
            if key in preview:
                parts.append(f"{key}={preview[key]}")
        if "input_chars" in preview:
            parts.append(f"input_chars={preview['input_chars']}")
        if "output_preview" in preview:
            parts.append(f"output={preview['output_preview']!r}")
        if "answer_preview" in preview:
            parts.append(f"answer={preview['answer_preview']!r}")
        parts.append(f"detail={event['detail_path']}")
        return " | ".join(parts)

    def _payload_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        preview: dict[str, Any] = {}
        for key in (
            "question",
            "agent_name",
            "round",
            "mode",
            "trigger",
            "error",
            "skipped",
            "ready_task_ids",
            "pending_task_ids",
            "completed_task_ids",
            "should_answer",
            "confidence",
            "passed",
            "score",
            "completed",
            "ready_for_final",
        ):
            if key in payload:
                preview[key] = payload[key]
        if "call" in payload and isinstance(payload["call"], dict):
            call = payload["call"]
            preview["tool_name"] = call.get("tool_name")
            preview["purpose"] = call.get("purpose")
            preview["args"] = call.get("args")
        if "output" in payload:
            preview["output_preview"] = self._shorten(payload["output"])
        if "answer" in payload:
            preview["answer_preview"] = self._shorten(payload["answer"])
        if "data" in payload and isinstance(payload["data"], dict):
            preview["json_keys"] = list(payload["data"].keys())
        if "system" in payload or "user" in payload:
            preview["input_chars"] = len(str(payload.get("system", ""))) + len(str(payload.get("user", "")))
        if "tasks" in payload:
            preview["task_ids"] = [task.get("id") for task in payload["tasks"] if isinstance(task, dict)]
        if "perspectives" in payload:
            preview["perspective_ids"] = [
                perspective.get("id") for perspective in payload["perspectives"] if isinstance(perspective, dict)
            ]
        return preview

    def _event_action(self, event_type: str) -> str:
        if event_type.endswith("_started"):
            return "START"
        if event_type.endswith("_finished"):
            return "DONE"
        if event_type == "llm_input":
            return "CALL"
        if event_type in {"llm_json_output", "component_result", "tool_result"}:
            return "RESULT"
        if event_type == "tool_call":
            return "TOOL"
        if event_type == "collaboration_selected":
            return "PLAN"
        return "EVENT"

    def _agent_log_path(self) -> Path:
        if self._event_path is None:
            raise RuntimeError("MAS event path is not configured.")
        name = self._event_path.name
        if name.endswith("-events.jsonl"):
            return self._event_path.with_name(name.replace("-events.jsonl", "-agent.log"))
        return self._event_path.with_suffix(".log")

    def _shorten(self, value: Any, limit: int = 240) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        return text if len(text) <= limit else text[:limit] + "...[truncated]"

    def _safe_name(self, value: str | None) -> str:
        if not value:
            return "none"
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:80]
