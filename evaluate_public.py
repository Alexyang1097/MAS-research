from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import traceback
from dataclasses import dataclass
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
from finance_mas.judge_backend import JudgeBackendConfig, OpenAICompatibleJudgeBackend


JUDGE_SYSTEM = """
You are a strict finance benchmark judge.
Evaluate the candidate answer against the provided rubric items.

Rubric operators:
- correctness: pass only if the answer explicitly states or strongly supports the criterion.
- contradiction: pass only if the answer does not contradict the criterion. If the answer is silent, pass unless silence creates a direct contradiction.

Return only JSON:
{
  "items": [
    {
      "index": 0,
      "operator": "correctness",
      "criteria": "string",
      "passed": true,
      "score": 1.0,
      "explanation": "short reason"
    }
  ],
  "score": 0.0,
  "passed_count": 0,
  "total_count": 0,
  "overall_explanation": "short summary"
}
""".strip()


@dataclass(frozen=True)
class PublicExample:
    index: int
    question: str
    reference_answer: str
    question_type: str
    rubric: list[dict[str, Any]]


def load_public_examples(path: Path, start_index: int, limit: int | None) -> list[PublicExample]:
    examples: list[PublicExample] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < start_index:
                continue
            rubric = json.loads(row["Rubric"])
            examples.append(
                PublicExample(
                    index=idx,
                    question=row["Question"],
                    reference_answer=row["Answer"],
                    question_type=row["Question Type"],
                    rubric=rubric,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def load_existing_results(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            existing[int(record["index"])] = record
    return existing


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_per_question_json(records: list[dict[str, Any]], output_dir: Path) -> Path:
    by_index = {record["index"]: record for record in records}
    records = [by_index[index] for index in sorted(by_index)]
    json_path = output_dir / "per_question.json"
    with json_path.open("w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return json_path


def refresh_per_question_json_from_jsonl(jsonl_path: Path) -> None:
    if not jsonl_path.exists():
        return
    write_per_question_json(list(load_existing_results(jsonl_path).values()), jsonl_path.parent)


def save_stage_record(path: Path, record: dict[str, Any], stage: str) -> None:
    record["last_stage"] = stage
    stages_completed = record.setdefault("stages_completed", [])
    if stage not in stages_completed:
        stages_completed.append(stage)
    append_jsonl(path, record)
    refresh_per_question_json_from_jsonl(path)


def _record_complete(record: dict[str, Any], systems: list[str]) -> bool:
    answers = record.get("answers", {})
    judgments = record.get("judgments", {})
    required_labels = ["reference", *systems]
    return all(label in answers and label in judgments for label in required_labels)


async def generate_baseline_answer(question: str, llm: Any, args: argparse.Namespace, question_id: str) -> str:
    parameters = Parameters(
        model_name=args.answer_model,
        max_turns=args.max_turns,
        tools=args.tools,
        llm_config=LLMConfig(max_tokens=args.max_tokens, temperature=args.temperature),
    )
    agent = get_agent(parameters, llm=llm)
    prompt = INSTRUCTIONS_PROMPT.format(question=question)
    result = await agent.run([TextInput(text=prompt)], question_id=question_id)
    return result.final_answer


async def generate_mas_answer(
    question: str,
    llm: Any,
    args: argparse.Namespace,
    mas_event_path: Path | None = None,
) -> str:
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
    runner = MASRunner(llm=llm, parameters=mas_parameters, event_path=mas_event_path)
    result = await runner.run(question)
    return result.final_answer


async def judge_answer(
    judge_backend: OpenAICompatibleJudgeBackend,
    *,
    question: str,
    answer: str,
    rubric: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    user = {
        "question": question,
        "answer_label": label,
        "candidate_answer": answer,
        "rubric": rubric,
    }
    result = await judge_backend.judge_json(system=JUDGE_SYSTEM, payload=user)
    result["label"] = label
    return result


async def process_example(
    example: PublicExample,
    *,
    answer_llm: Any,
    judge_backend: OpenAICompatibleJudgeBackend,
    args: argparse.Namespace,
    per_question_path: Path,
    existing_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = existing_record or {
        "index": example.index,
        "question": example.question,
        "question_type": example.question_type,
        "answers": {},
        "judgments": {},
        "stages_completed": [],
    }
    answers: dict[str, str] = record.setdefault("answers", {})
    judgments: dict[str, Any] = record.setdefault("judgments", {})
    stages_completed: list[str] = record.setdefault("stages_completed", [])

    if "reference" not in answers:
        answers["reference"] = example.reference_answer
        save_stage_record(per_question_path, record, "reference_answer_saved")

    if "reference" not in judgments:
        judgments["reference"] = await judge_answer(
            judge_backend,
            question=example.question,
            answer=answers["reference"],
            rubric=example.rubric,
            label="reference",
        )
        save_stage_record(per_question_path, record, "reference_judged")

    # Run MAS before the finance-agent baseline so MAS progress is preserved even if baseline is slow.
    if "mas" in args.systems and "mas" not in answers:
        mas_event_path = args.output_dir / "mas_events" / f"public-{example.index:03d}-mas-events.jsonl"
        artifacts = record.setdefault("artifacts", {})
        artifacts["mas_events"] = str(mas_event_path)
        artifacts["mas_agent_log"] = str(mas_event_path.with_name(f"public-{example.index:03d}-mas-agent.log"))
        artifacts["mas_event_details_dir"] = str(mas_event_path.with_name(f"public-{example.index:03d}-mas-events_details"))
        save_stage_record(per_question_path, record, "mas_started")
        answers["mas"] = await generate_mas_answer(
            example.question,
            answer_llm,
            args,
            mas_event_path=mas_event_path,
        )
        save_stage_record(per_question_path, record, "mas_answered")

    if "mas" in answers and "mas" not in judgments:
        judgments["mas"] = await judge_answer(
            judge_backend,
            question=example.question,
            answer=answers["mas"],
            rubric=example.rubric,
            label="mas",
        )
        save_stage_record(per_question_path, record, "mas_judged")

    if "baseline" in args.systems and "baseline" not in answers:
        save_stage_record(per_question_path, record, "baseline_started")
        answers["baseline"] = await generate_baseline_answer(
            example.question,
            answer_llm,
            args,
            question_id=f"public-{example.index:03d}-baseline",
        )
        save_stage_record(per_question_path, record, "baseline_answered")

    if "baseline" in answers and "baseline" not in judgments:
        judgments["baseline"] = await judge_answer(
            judge_backend,
            question=example.question,
            answer=answers["baseline"],
            rubric=example.rubric,
            label="baseline",
        )
        save_stage_record(per_question_path, record, "baseline_judged")

    save_stage_record(per_question_path, record, "complete")
    return record


def score_of(record: dict[str, Any], label: str) -> float | None:
    judgment = record.get("judgments", {}).get(label)
    if not judgment:
        return None
    return float(judgment.get("score", 0.0))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({label for record in records for label in record.get("judgments", {})})
    summary: dict[str, Any] = {
        "num_questions": len(records),
        "labels": labels,
        "average_scores": {},
        "pass_rates": {},
    }
    for label in labels:
        scores = [score_of(record, label) for record in records]
        scores = [score for score in scores if score is not None]
        summary["average_scores"][label] = sum(scores) / len(scores) if scores else None
        passed = [
            bool(record["judgments"][label].get("passed_count", 0) == record["judgments"][label].get("total_count", 1))
            for record in records
            if label in record.get("judgments", {})
        ]
        summary["pass_rates"][label] = sum(passed) / len(passed) if passed else None

    if "mas" in labels and "baseline" in labels:
        mas_wins = baseline_wins = ties = comparable = 0
        for record in records:
            mas_score = score_of(record, "mas")
            baseline_score = score_of(record, "baseline")
            if mas_score is None or baseline_score is None:
                continue
            comparable += 1
            if mas_score > baseline_score:
                mas_wins += 1
            elif baseline_score > mas_score:
                baseline_wins += 1
            else:
                ties += 1
        summary["mas_vs_baseline"] = {
            "comparable": comparable,
            "mas_wins": mas_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "mas_win_rate_excluding_ties": mas_wins / (mas_wins + baseline_wins)
            if (mas_wins + baseline_wins)
            else None,
            "mas_non_loss_rate": (mas_wins + ties) / comparable if comparable else None,
        }

    if "reference" in labels:
        summary["reference_average_score"] = summary["average_scores"].get("reference")
    return summary


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    by_index = {record["index"]: record for record in records}
    records = [by_index[index] for index in sorted(by_index)]
    summary = summarize(records)
    per_question_json_path = write_per_question_json(records, output_dir)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "scores.csv"
    labels = summary["labels"]
    fieldnames = ["index", "question", "last_stage", "error", *[f"{label}_score" for label in labels]]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "index": record["index"],
                "question": record["question"],
                "last_stage": record.get("last_stage"),
                "error": record.get("error"),
            }
            for label in labels:
                row[f"{label}_score"] = score_of(record, label)
            writer.writerow(row)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Per-question JSON: {per_question_json_path}")
    print(f"Summary: {summary_path}")
    print(f"Scores CSV: {csv_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and MAS on finance-agent/data/public.csv.")
    parser.add_argument("--public-csv", type=Path, default=FINANCE_AGENT_ROOT / "data" / "public.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "public_eval")
    parser.add_argument("--answer-model", default="deepseek/deepseek-reasoner")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-fallback-models", nargs="*", default=["deepseek-reasoner", "deepseek-chat"])
    parser.add_argument("--judge-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--systems", nargs="+", default=["baseline", "mas"], choices=["baseline", "mas"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parallelism", type=int, default=1)
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv(override=True, dotenv_path=FINANCE_AGENT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for rubric judging.")

    examples = load_public_examples(args.public_csv, args.start_index, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_question_path = args.output_dir / "per_question.jsonl"
    existing = load_existing_results(per_question_path) if args.resume else {}

    answer_llm = get_registry_model(args.answer_model, LLMConfig(max_tokens=args.max_tokens, temperature=args.temperature))
    judge_backend = OpenAICompatibleJudgeBackend(
        JudgeBackendConfig(
            api_key=api_key,
            base_url=args.judge_base_url,
            model=args.judge_model,
            fallback_models=args.judge_fallback_models,
            temperature=0.0,
        )
    )
    semaphore = asyncio.Semaphore(args.parallelism)
    completed: list[dict[str, Any]] = list(existing.values())
    finished_this_run: list[dict[str, Any]] = []

    async def run_one(example: PublicExample):
        existing_record = existing.get(example.index)
        if existing_record is not None and _record_complete(existing_record, args.systems):
            return existing_record
        async with semaphore:
            try:
                record = await process_example(
                    example,
                    answer_llm=answer_llm,
                    judge_backend=judge_backend,
                    args=args,
                    per_question_path=per_question_path,
                    existing_record=existing_record,
                )
            except Exception as error:
                record = existing_record or {
                    "index": example.index,
                    "question": example.question,
                    "question_type": example.question_type,
                    "answers": {},
                    "judgments": {},
                    "stages_completed": [],
                }
                record["error"] = f"{type(error).__name__}: {error}"
                record["traceback"] = traceback.format_exc()
                save_stage_record(per_question_path, record, "failed")
                finished_this_run.append(record)
                write_outputs(completed + finished_this_run, args.output_dir)
                print(f"Failed #{example.index}: {record['error']}")
                return record

            finished_this_run.append(record)
            write_outputs(completed + finished_this_run, args.output_dir)
            print(
                f"Completed #{example.index}: "
                f"baseline={score_of(record, 'baseline')} mas={score_of(record, 'mas')} reference={score_of(record, 'reference')}"
            )
            return record

    new_records = await asyncio.gather(*[run_one(example) for example in examples], return_exceptions=False)
    write_outputs(completed + new_records, args.output_dir)
    print(f"Per-question results: {per_question_path}")


if __name__ == "__main__":
    asyncio.run(main())
