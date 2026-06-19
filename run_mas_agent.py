from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
FINANCE_AGENT_ROOT = REPO_ROOT / "finance-agent"
sys.path.insert(0, str(FINANCE_AGENT_ROOT))

from dotenv import load_dotenv
from model_library.base import LLMConfig
from model_library.registry_utils import get_registry_model
from tqdm.asyncio import tqdm

from finance_agent.tools import VALID_TOOLS
from finance_mas import MASParameters, MASRunner


async def run_tests_parallel(
    questions: list[str],
    max_concurrent: int,
    model_name: str,
    llm_config: LLMConfig,
    mas_parameters: MASParameters,
):
    semaphore = asyncio.Semaphore(max_concurrent)
    llm = get_registry_model(model_name, llm_config)

    async def process_question(question: str):
        async with semaphore:
            runner = MASRunner(llm=llm, parameters=mas_parameters)
            return await runner.run(question)

    results = await tqdm.gather(
        *[process_question(question) for question in questions],
        desc="Processing MAS questions",
    )

    formatted_results = []
    for result in results:
        formatted_results.append(result.model_dump(mode="json"))
        status = "OK" if result.success else "FAIL"
        print(
            f"\n{status} MAS question: {result.question}\n"
            f"   Iterations: {result.iterations}\n"
            f"   Rubric failures: {result.rubric_failures}\n"
            f"   Result: {result.final_answer}\n"
        )

    output_dir = Path("logs") / "mas"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(formatted_results, f, indent=2)
    print(f"\nMAS results saved to: {output_file}")
    return formatted_results


async def main():
    parser = argparse.ArgumentParser(description="Run the MAS harness for the finance agent benchmark")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--questions", type=str, nargs="+")
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--model", type=str, default="deepseek/deepseek-reasoner")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Hard safety cap for total orchestration loops; normal stopping is controlled by final-goal attempts",
    )
    parser.add_argument(
        "--max-final-goal-replans",
        type=int,
        default=3,
        help="Maximum replans allowed after the Decider says the next goal is the final goal",
    )
    parser.add_argument("--max-rubric-failures", type=int, default=2)
    parser.add_argument("--debate-rounds-cap", type=int, default=2)
    parser.add_argument("--max-plan-refinements", type=int, default=1)
    parser.add_argument("--max-tool-rounds-per-subagent", type=int, default=3)
    parser.add_argument("--max-tool-calls-per-round", type=int, default=3)
    parser.add_argument("--tools", type=str, nargs="+", default=VALID_TOOLS, choices=VALID_TOOLS)
    args = parser.parse_args()

    load_dotenv(override=True, dotenv_path=FINANCE_AGENT_ROOT / ".env")

    if args.question_file:
        with open(args.question_file) as f:
            questions = [line.strip() for line in f if line.strip()]
    elif args.questions:
        questions = args.questions
    else:
        raise Exception("No questions provided. One of --question-file or --questions must be used.")

    await run_tests_parallel(
        questions=questions,
        max_concurrent=args.parallelism,
        model_name=args.model,
        llm_config=LLMConfig(max_tokens=args.max_tokens, temperature=args.temperature),
        mas_parameters=MASParameters(
            max_iterations=args.max_iterations,
            max_final_goal_replans=args.max_final_goal_replans,
            max_rubric_failures=args.max_rubric_failures,
            max_tool_rounds_per_subagent=args.max_tool_rounds_per_subagent,
            max_tool_calls_per_round=args.max_tool_calls_per_round,
            debate_rounds_cap=args.debate_rounds_cap,
            max_plan_refinements=args.max_plan_refinements,
            tools=args.tools,
        ),
    )


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
