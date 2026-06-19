from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
FINANCE_AGENT_ROOT = REPO_ROOT / "finance-agent"
sys.path.insert(0, str(FINANCE_AGENT_ROOT))

from model_library.base import LLMConfig
from model_library.base.input import TextInput
from model_library.registry_utils import get_registry_model

from finance_agent.get_agent import Parameters, get_agent
from finance_agent.prompt import INSTRUCTIONS_PROMPT
from finance_agent.tools import VALID_TOOLS
from finance_mas import MASParameters, MASRunner


def create_llm_config(args: argparse.Namespace) -> LLMConfig:
    return LLMConfig(max_tokens=args.max_tokens, temperature=args.temperature)


async def create_baseline_model_func(args: argparse.Namespace):
    llm_config = create_llm_config(args)
    llm = get_registry_model(args.model, llm_config)
    parameters = Parameters(
        model_name=args.model,
        max_turns=args.max_turns,
        tools=args.tools,
        llm_config=llm_config,
    )

    async def model_func(test_input: str, files: dict[str, Any] | None = None, context: dict[str, Any] | None = None):
        agent = get_agent(parameters, llm=llm)
        prompt = INSTRUCTIONS_PROMPT.format(question=test_input)
        question_id = str((context or {}).get("question_id", "vals-baseline"))
        result = await agent.run([TextInput(text=prompt)], question_id=question_id)
        return result.final_answer

    return model_func


async def create_mas_model_func(args: argparse.Namespace):
    llm_config = create_llm_config(args)
    llm = get_registry_model(args.model, llm_config)
    mas_parameters = MASParameters(
        max_iterations=args.max_iterations,
        max_final_goal_replans=args.max_final_goal_replans,
        max_rubric_failures=args.max_rubric_failures,
        max_tool_rounds_per_subagent=args.max_tool_rounds_per_subagent,
        max_tool_calls_per_round=args.max_tool_calls_per_round,
        debate_rounds_cap=args.debate_rounds_cap,
        max_plan_refinements=args.max_plan_refinements,
        tools=args.tools,
    )

    async def model_func(test_input: str, files: dict[str, Any] | None = None, context: dict[str, Any] | None = None):
        runner = MASRunner(llm=llm, parameters=mas_parameters)
        result = await runner.run(test_input)
        return result.final_answer

    return model_func


async def main():
    parser = argparse.ArgumentParser(description="Run a Vals test suite with the baseline finance agent or MAS.")
    parser.add_argument("--suite-id", required=True, help="Vals test suite ID.")
    parser.add_argument("--mode", choices=["baseline", "mas"], default="mas")
    parser.add_argument("--model", default="deepseek/deepseek-reasoner")
    parser.add_argument("--model-name", help="Display name for the run in Vals. Defaults to mode + model.")
    parser.add_argument("--run-name")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--eval-model")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-final-goal-replans", type=int, default=3)
    parser.add_argument("--max-rubric-failures", type=int, default=2)
    parser.add_argument("--debate-rounds-cap", type=int, default=2)
    parser.add_argument("--max-plan-refinements", type=int, default=1)
    parser.add_argument("--max-tool-rounds-per-subagent", type=int, default=3)
    parser.add_argument("--max-tool-calls-per-round", type=int, default=3)
    parser.add_argument("--tools", nargs="+", default=VALID_TOOLS, choices=VALID_TOOLS)
    args = parser.parse_args()

    load_dotenv(override=True, dotenv_path=FINANCE_AGENT_ROOT / ".env")

    from vals import RunParameters, Suite

    suite = await Suite.from_id(args.suite_id)
    model_func = await (create_mas_model_func(args) if args.mode == "mas" else create_baseline_model_func(args))
    model_name = args.model_name or f"finance-{args.mode}-{args.model}"

    run_parameters_kwargs: dict[str, Any] = {"parallelism": args.parallelism}
    if args.eval_model:
        run_parameters_kwargs["eval_model"] = args.eval_model

    run = await suite.run(
        model=model_func,
        model_name=model_name,
        run_name=args.run_name,
        wait_for_completion=True,
        parameters=RunParameters(**run_parameters_kwargs),
        except_on_error=True,
    )

    print(f"Run URL: {run.url}")
    print(f"Status: {run.status}")
    print(f"Pass rate: {run.pass_rate}")
    print(f"Success rate: {run.success_rate}")


if __name__ == "__main__":
    asyncio.run(main())
