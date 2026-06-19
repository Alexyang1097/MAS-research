from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from model_library.agent import Tool, ToolOutput
from model_library.base import LLM

from .llm_utils import query_json
from .prompts import CRITIQUE_SYSTEM, subagent_system_for
from .schemas import CritiqueOutput, SubagentOutput, ToolCallRequest
from .schema_utils import fallback_subagent_output, parse_critique_output, parse_subagent_output
from .subagent_registry import default_tools_for

EventRecorder = Callable[..., None]


class ToolExecutor:
    def __init__(self, tools: list[Tool], state: dict[str, Any], logger: logging.Logger):
        self._tools = {tool.name: tool for tool in tools}
        self._state = state
        self._logger = logger

    @property
    def state_keys(self) -> list[str]:
        return list(self._state.keys())

    def tool_specs(self, allowed_tools: list[str]) -> list[dict[str, Any]]:
        specs = []
        for tool_name in allowed_tools:
            tool = self._tools.get(tool_name)
            if tool is None:
                continue
            specs.append(
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                    "required": getattr(tool, "required", []),
                }
            )
        return specs

    async def execute(self, call: ToolCallRequest) -> ToolOutput:
        call = self.normalize_call(call)
        if call.tool_name == "retrieve_information" and "prompt" not in call.args:
            repaired = await self._repair_retrieve_information_call(call)
            if repaired is not None:
                return repaired
            available_keys = ", ".join(self._state.keys()) or "(none)"
            return ToolOutput(
                output=(
                    "ERROR: retrieve_information requires args {'prompt': '... {{stored_key}} ...'} and can only "
                    f"query documents already saved in data storage. Available keys: {available_keys}. "
                    "If you only have a URL, call parse_html_page with {'url': ..., 'key': ...} first."
                ),
                error="retrieve_information missing required prompt",
            )

        tool = self._tools.get(call.tool_name)
        if tool is None:
            return ToolOutput(
                output=f"Unknown tool '{call.tool_name}'. Available tools: {sorted(self._tools)}",
                error=f"Unknown tool '{call.tool_name}'",
            )
        return await tool.execute(call.args, self._state, self._logger)

    def normalize_call(self, call: ToolCallRequest) -> ToolCallRequest:
        args = dict(call.args)
        tool_name = call.tool_name

        if tool_name == "web_search":
            if "search_query" not in args:
                for alias in ("query", "q", "keywords"):
                    if alias in args:
                        args["search_query"] = args.pop(alias)
                        break
            return call.model_copy(update={"args": args})

        if tool_name == "edgar_search":
            if "search_query" not in args:
                for alias in ("query", "q", "keywords"):
                    if alias in args:
                        args["search_query"] = args.pop(alias)
                        break
            if isinstance(args.get("form_types"), str):
                args["form_types"] = [args["form_types"]]
            if isinstance(args.get("ciks"), str):
                args["ciks"] = [args["ciks"]]
            return call.model_copy(update={"args": args})

        if tool_name == "parse_html_page":
            if "url" not in args:
                for alias in ("source_url", "link", "href"):
                    if alias in args:
                        args["url"] = args.pop(alias)
                        break
            if "key" not in args and isinstance(args.get("url"), str):
                args["key"] = self._storage_key_from_url(args["url"])
            return call.model_copy(update={"args": args})

        if tool_name == "retrieve_information":
            if "prompt" not in args:
                for alias in ("query", "question", "instruction"):
                    if alias in args:
                        args["prompt"] = args.pop(alias)
                        break
            key = args.get("key")
            if "prompt" in args and isinstance(key, str) and "{{" not in str(args["prompt"]):
                args["prompt"] = f"{args['prompt']}: {{{{{key}}}}}"
            return call.model_copy(update={"args": args})

        return call

    async def _repair_retrieve_information_call(self, call: ToolCallRequest) -> ToolOutput | None:
        url = call.args.get("url")
        parse_tool = self._tools.get("parse_html_page")
        if not isinstance(url, str) or not url.strip() or parse_tool is None:
            return None

        key = call.args.get("key")
        if not isinstance(key, str) or not key.strip():
            key = self._storage_key_from_url(url)

        parse_output = await parse_tool.execute({"url": url, "key": key}, self._state, self._logger)
        if parse_output.error:
            return ToolOutput(
                output=(
                    "The requested retrieve_information call used a URL instead of a stored-key prompt, so it was "
                    f"converted to parse_html_page first, but parsing failed.\nURL: {url}\nError: {parse_output.error}"
                ),
                error=parse_output.error,
                metadata=parse_output.metadata,
            )

        return ToolOutput(
            output=(
                "The requested retrieve_information call used a URL instead of a stored-key prompt. "
                "I converted it to parse_html_page and saved the page first.\n"
                f"{parse_output.output}\n"
                f"Next call retrieve_information with args: "
                f"{{'prompt': 'Answer the assigned question using this stored document: {{{{{key}}}}}'}}"
            ),
            metadata=parse_output.metadata,
        )

    def _storage_key_from_url(self, url: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_").lower()
        slug = slug[-48:] or "parsed_source"
        key = f"auto_{slug}"
        if key not in self._state:
            return key

        suffix = 2
        while f"{key}_{suffix}" in self._state:
            suffix += 1
        return f"{key}_{suffix}"


class SubagentRunner:
    def __init__(
        self,
        llm: LLM,
        tool_executor: ToolExecutor,
        logger: logging.Logger,
        max_tool_rounds: int = 3,
        max_tool_calls_per_round: int = 3,
        event_recorder: EventRecorder | None = None,
    ):
        self._llm = llm
        self._tool_executor = tool_executor
        self._logger = logger
        self._max_tool_rounds = max_tool_rounds
        self._max_tool_calls_per_round = max_tool_calls_per_round
        self._event_recorder = event_recorder

    async def run_assignment(
        self,
        *,
        agent_name: str,
        assignment_id: str,
        goal: str,
        global_context: str,
        allowed_tools: list[str] | None = None,
        extra_instruction: str | None = None,
    ) -> SubagentOutput:
        allowed = allowed_tools or default_tools_for(agent_name)
        observations: list[dict[str, Any]] = []
        base_user = self._build_user_prompt(
            agent_name=agent_name,
            assignment_id=assignment_id,
            goal=goal,
            global_context=global_context,
            allowed_tools=allowed,
            extra_instruction=extra_instruction,
        )
        system_prompt = subagent_system_for(agent_name)
        self._record_event(
            "assignment_started",
            assignment_id,
            {
                "agent_name": agent_name,
                "goal": goal,
                "allowed_tools": allowed,
                "global_context": global_context,
                "extra_instruction": extra_instruction,
                "system": system_prompt,
                "base_user": base_user,
            },
        )

        for _round_index in range(self._max_tool_rounds + 1):
            user = base_user
            if observations:
                user += "\n\nTool observations so far:\n" + json.dumps(observations, ensure_ascii=False, default=str)
            self._record_event(
                "llm_input",
                assignment_id,
                {
                    "agent_name": agent_name,
                    "round": _round_index,
                    "system": system_prompt,
                    "user": user,
                },
            )
            try:
                data = await query_json(self._llm, system_prompt, user)
            except Exception as error:
                output = fallback_subagent_output(
                    agent_name=agent_name,
                    assignment_id=assignment_id,
                    goal=goal,
                    error=error,
                )
                self._record_event(
                    "assignment_failed",
                    assignment_id,
                    {"agent_name": agent_name, "round": _round_index, "error": str(error), "output": output.model_dump(mode="json")},
                )
                return output
            self._record_event(
                "llm_json_output",
                assignment_id,
                {"agent_name": agent_name, "round": _round_index, "data": data},
            )
            if data.get("needs_tools"):
                tool_calls = []
                for raw_call in data.get("tool_calls", [])[: self._max_tool_calls_per_round]:
                    try:
                        tool_calls.append(ToolCallRequest.model_validate(raw_call))
                    except Exception as error:
                        observations.append(
                            {
                                "tool": "invalid_tool_call",
                                "purpose": "schema recovery",
                                "error": str(error),
                                "raw_call": raw_call,
                            }
                        )
                if not tool_calls:
                    break
                for call in tool_calls:
                    call = self._tool_executor.normalize_call(call)
                    self._record_event(
                        "tool_call",
                        assignment_id,
                        {"agent_name": agent_name, "round": _round_index, "call": call.model_dump(mode="json")},
                    )
                    if call.tool_name not in allowed:
                        observations.append(
                            {
                                "tool": call.tool_name,
                                "purpose": call.purpose,
                                "error": f"Tool not allowed for this assignment. Allowed: {allowed}",
                            }
                        )
                        self._record_event(
                            "tool_result",
                            assignment_id,
                            {"agent_name": agent_name, "round": _round_index, "call": call.model_dump(mode="json"), "error": observations[-1]["error"]},
                        )
                        continue
                    if self._is_repeated_failed_call(call, observations):
                        observations.append(
                            {
                                "tool": call.tool_name,
                                "purpose": call.purpose,
                                "args": call.args,
                                "error": "Repeated identical failed tool call skipped.",
                                "output": (
                                    "This exact tool call already failed. Do not retry it; use another source/tool "
                                    "or produce the best supported partial answer with this limitation noted."
                                ),
                            }
                        )
                        self._record_event(
                            "tool_result",
                            assignment_id,
                            {"agent_name": agent_name, "round": _round_index, "call": call.model_dump(mode="json"), "skipped": True, "error": observations[-1]["error"], "output": observations[-1]["output"]},
                        )
                        continue
                    output = await self._tool_executor.execute(call)
                    observations.append(
                        {
                            "tool": call.tool_name,
                            "purpose": call.purpose,
                            "args": call.args,
                            "output": output.output,
                            "error": output.error,
                        }
                    )
                    self._record_event(
                        "tool_result",
                        assignment_id,
                        {
                            "agent_name": agent_name,
                            "round": _round_index,
                            "call": call.model_dump(mode="json"),
                            "output": output.output,
                            "error": output.error,
                        },
                    )
                continue

            data["agent_name"] = agent_name
            data["assignment_id"] = assignment_id
            data["goal"] = goal
            try:
                output = parse_subagent_output(data, agent_name=agent_name, assignment_id=assignment_id, goal=goal)
            except Exception as error:
                output = fallback_subagent_output(
                    agent_name=agent_name,
                    assignment_id=assignment_id,
                    goal=goal,
                    error=error,
                )
            self._record_event(
                "assignment_finished",
                assignment_id,
                {"agent_name": agent_name, "round": _round_index, "output": output.model_dump(mode="json")},
            )
            return output

        fallback_user = self._build_fallback_prompt(base_user, observations)
        self._record_event(
            "llm_input",
            assignment_id,
            {"agent_name": agent_name, "round": "fallback", "system": system_prompt, "user": fallback_user},
        )
        try:
            data = await query_json(self._llm, system_prompt, fallback_user)
        except Exception as error:
            output = fallback_subagent_output(
                agent_name=agent_name,
                assignment_id=assignment_id,
                goal=goal,
                error=error,
            )
            self._record_event(
                "assignment_failed",
                assignment_id,
                {"agent_name": agent_name, "round": "fallback", "error": str(error), "output": output.model_dump(mode="json")},
            )
            return output
        self._record_event(
            "llm_json_output",
            assignment_id,
            {"agent_name": agent_name, "round": "fallback", "data": data},
        )
        data["needs_tools"] = False
        data["agent_name"] = agent_name
        data["assignment_id"] = assignment_id
        data["goal"] = goal
        try:
            output = parse_subagent_output(data, agent_name=agent_name, assignment_id=assignment_id, goal=goal)
        except Exception as error:
            output = fallback_subagent_output(
                agent_name=agent_name,
                assignment_id=assignment_id,
                goal=goal,
                error=error,
            )
        if output.answer in {"No answer provided.", "No substantive answer provided."}:
            output = self._observation_based_output(
                agent_name=agent_name,
                assignment_id=assignment_id,
                goal=goal,
                observations=observations,
            )
        self._record_event(
            "assignment_finished",
            assignment_id,
            {"agent_name": agent_name, "round": "fallback", "output": output.model_dump(mode="json")},
        )
        return output

    async def critique(
        self,
        *,
        agent_name: str,
        global_context: str,
        own_output: SubagentOutput,
        all_outputs: list[SubagentOutput],
    ) -> CritiqueOutput:
        user = (
            f"Agent: {agent_name}\n\nGlobal context:\n{global_context}\n\n"
            f"Your original output:\n{own_output.model_dump_json(indent=2)}\n\n"
            "All candidate outputs:\n"
            + json.dumps([item.model_dump(mode="json") for item in all_outputs], indent=2)
        )
        self._record_event("llm_input", own_output.assignment_id, {"agent_name": agent_name, "component": "critique", "system": CRITIQUE_SYSTEM, "user": user})
        data = await query_json(self._llm, CRITIQUE_SYSTEM, user)
        self._record_event("llm_json_output", own_output.assignment_id, {"agent_name": agent_name, "component": "critique", "data": data})
        critique = parse_critique_output(data, agent_name=agent_name)
        self._record_event("critique_finished", own_output.assignment_id, {"agent_name": agent_name, "output": critique.model_dump(mode="json")})
        return critique

    def _build_user_prompt(
        self,
        *,
        agent_name: str,
        assignment_id: str,
        goal: str,
        global_context: str,
        allowed_tools: list[str],
        extra_instruction: str | None,
    ) -> str:
        parts = [
            f"Agent name: {agent_name}",
            f"Assignment id: {assignment_id}",
            f"Goal: {goal}",
            f"Allowed tools: {', '.join(allowed_tools) or '(none)'}",
            f"Current data storage keys: {', '.join(self._tool_executor.state_keys) or '(none)'}",
            "Allowed tool schemas:\n"
            + json.dumps(self._tool_executor.tool_specs(allowed_tools), indent=2, ensure_ascii=False),
            f"Global task context:\n{global_context}",
        ]
        if extra_instruction:
            parts.append(f"Extra instruction:\n{extra_instruction}")
        return "\n\n".join(parts)

    def _build_fallback_prompt(self, base_user: str, observations: list[dict[str, Any]]) -> str:
        successful_observations = [item for item in observations if item.get("error") in {None, ""}]
        failed_observations = [item for item in observations if item.get("error") not in {None, ""}]
        return (
            base_user
            + "\n\nYou have reached the tool-call limit. You MUST stop requesting tools and produce the best possible final JSON now.\n"
            + "Do not return an empty answer. Base your answer on the successful tool observations below.\n"
            + "Your final JSON must include:\n"
            + "- answer: a concise synthesis of facts found in observations, including dates, numbers, source names/URLs, and caveats.\n"
            + "- evidence: one item per important supported claim, using source_url when present in tool outputs.\n"
            + "- calculations: formulas or numeric checks if any numbers were found.\n"
            + "- assumptions/open_questions/failure_modes: include failed retrievals or missing evidence, but do not let failures erase successful findings.\n\n"
            + "Successful tool observations to synthesize:\n"
            + json.dumps(successful_observations, ensure_ascii=False, default=str)
            + "\n\nFailed or skipped tool observations to mention as limitations:\n"
            + json.dumps(failed_observations, ensure_ascii=False, default=str)
        )

    def _observation_based_output(
        self,
        *,
        agent_name: str,
        assignment_id: str,
        goal: str,
        observations: list[dict[str, Any]],
    ) -> SubagentOutput:
        successful = [item for item in observations if item.get("error") in {None, ""}]
        failed = [item for item in observations if item.get("error") not in {None, ""}]
        if not successful:
            return SubagentOutput(
                agent_name=agent_name,
                assignment_id=assignment_id,
                goal=goal,
                answer="No substantive answer provided and no successful tool observations were available.",
                open_questions=[item.get("error", "unknown error") for item in failed],
                failure_modes=["tool_limit_reached_without_successful_observations"],
                confidence=0.0,
            )
        summarized = []
        for item in successful:
            summarized.append(
                {
                    "tool": item.get("tool"),
                    "purpose": item.get("purpose"),
                    "args": item.get("args"),
                    "output": item.get("output"),
                }
            )
        return SubagentOutput(
            agent_name=agent_name,
            assignment_id=assignment_id,
            goal=goal,
            answer=(
                "Tool-call limit reached before a schema-complete synthesis was produced. "
                "Successful observations are preserved for aggregation: "
                + json.dumps(summarized, ensure_ascii=False, default=str)
            ),
            open_questions=[item.get("error", "unknown error") for item in failed],
            failure_modes=["fallback_used_successful_tool_observations"],
            confidence=0.45,
        )

    def _is_repeated_failed_call(self, call: ToolCallRequest, observations: list[dict[str, Any]]) -> bool:
        for observation in reversed(observations):
            if not observation.get("error"):
                continue
            if observation.get("tool") != call.tool_name:
                continue
            if observation.get("args") == call.args:
                return True
        return False

    def _record_event(self, event_type: str, assignment_id: str, payload: dict[str, Any]) -> None:
        if self._event_recorder is not None:
            self._event_recorder(
                event_type,
                component="subagent",
                assignment_id=assignment_id,
                payload=payload,
            )
