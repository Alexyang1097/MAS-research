from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from model_library.base import LLM
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    loaded = _loads_json_object(stripped)
    if loaded is not None:
        return loaded

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM response did not contain a JSON object: {text[:500]}")
    loaded = _loads_json_object(stripped[start : end + 1])
    if loaded is not None:
        return loaded

    raise ValueError(f"LLM response contained malformed JSON object: {text[:500]}")


def _loads_json_object(candidate: str) -> dict[str, Any] | None:
    for raw in (candidate, _escape_control_chars_in_json_strings(candidate)):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
        raise ValueError("LLM JSON response must be an object")
    return None


def _escape_control_chars_in_json_strings(text: str) -> str:
    """Escape raw control characters that models sometimes emit inside JSON strings."""
    output: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if escaped:
            output.append(char)
            escaped = False
            continue

        if char == "\\":
            output.append(char)
            escaped = in_string
            continue

        if char == '"':
            output.append(char)
            in_string = not in_string
            continue

        if in_string:
            if char == "\n":
                output.append("\\n")
                continue
            if char == "\r":
                output.append("\\r")
                continue
            if char == "\t":
                output.append("\\t")
                continue
            if ord(char) < 0x20:
                output.append(f"\\u{ord(char):04x}")
                continue

        output.append(char)

    return "".join(output)


async def query_text(llm: LLM, system: str, user: str) -> str:
    prompt = f"{system}\n\nUser/context:\n{user}"
    response = await llm.query(prompt)
    return response.output_text_str


async def query_json(llm: LLM, system: str, user: str) -> dict[str, Any]:
    return extract_json_object(await query_text(llm, system, user))


async def query_model(llm: LLM, model_type: type[ModelT], system: str, user: str) -> ModelT:
    data = await query_json(llm, system, user)
    return model_type.model_validate(data)
