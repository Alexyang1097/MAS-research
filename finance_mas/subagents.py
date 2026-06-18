from __future__ import annotations

import json
import logging
from typing import Any

from model_library.agent import Tool, ToolOutput
from model_library.base import LLM

from .llm_utils import query_json, query_model
from .prompts import CRITIQUE_SYSTEM, SUBAGENT_SYSTEM
from .schemas import CritiqueOutput, SubagentOutput, ToolCallRequest
from .subagent_registry import default_tools_for


class ToolExecutor:
    def __init__(self, tools: list[Tool], state: dict[str, Any], logger: logging.Logger):
        self._tools = {tool.name: tool for tool in tools}
        self._state = state
        self._logger = logger

    @property
    def state_keys(self) -> list[str]:
        return list(self._state.keys())

    async def execute(self, call: ToolCallRequest) -> ToolOutput:
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return ToolOutput(
                output=f"Unknown tool '{call.tool_name}'. Available tools: {sorted(self._tools)}",
                error=f"Unknown tool '{call.tool_name}'",
            )
        return await tool.execute(call.args, self._state, self._logger)


class SubagentRunner:
    def __init__(
        self,
        llm: LLM,
        tool_executor: ToolExecutor,
        logger: logging.Logger,
        max_tool_rounds: int = 3,
        max_tool_calls_per_round: int = 3,
    ):
        self._llm = llm
        self._tool_executor = tool_executor
        self._logger = logger
        self._max_tool_rounds = max_tool_rounds
        self._max_tool_calls_per_round = max_tool_calls_per_round

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

        for _round_index in range(self._max_tool_rounds + 1):
            user = base_user
            if observations:
                user += "\n\nTool observations so far:\n" + json.dumps(observations, ensure_ascii=False, default=str)
            data = await query_json(self._llm, SUBAGENT_SYSTEM, user)
            if data.get("needs_tools"):
                tool_calls = [
                    ToolCallRequest.model_validate(raw_call)
                    for raw_call in data.get("tool_calls", [])[: self._max_tool_calls_per_round]
                ]
                if not tool_calls:
                    break
                for call in tool_calls:
                    if call.tool_name not in allowed:
                        observations.append(
                            {
                                "tool": call.tool_name,
                                "purpose": call.purpose,
                                "error": f"Tool not allowed for this assignment. Allowed: {allowed}",
                            }
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
                continue

            data["agent_name"] = agent_name
            data["assignment_id"] = assignment_id
            data["goal"] = goal
            return SubagentOutput.model_validate(data)

        fallback_user = (
            base_user
            + "\n\nYou have reached the tool-call limit. Produce the best possible final JSON now.\n"
            + json.dumps(observations, ensure_ascii=False, default=str)
        )
        data = await query_json(self._llm, SUBAGENT_SYSTEM, fallback_user)
        data["needs_tools"] = False
        data["agent_name"] = agent_name
        data["assignment_id"] = assignment_id
        data["goal"] = goal
        return SubagentOutput.model_validate(data)

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
        result = await query_model(self._llm, CritiqueOutput, CRITIQUE_SYSTEM, user)
        result.agent_name = agent_name
        return result

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
            f"Global task context:\n{global_context}",
        ]
        if extra_instruction:
            parts.append(f"Extra instruction:\n{extra_instruction}")
        return "\n\n".join(parts)
