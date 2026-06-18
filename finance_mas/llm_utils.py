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
    try:
        loaded = json.loads(stripped)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM response did not contain a JSON object: {text[:500]}")
    loaded = json.loads(stripped[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("LLM JSON response must be an object")
    return loaded


async def query_text(llm: LLM, system: str, user: str) -> str:
    prompt = f"{system}\n\nUser/context:\n{user}"
    response = await llm.query(prompt)
    return response.output_text_str


async def query_json(llm: LLM, system: str, user: str) -> dict[str, Any]:
    return extract_json_object(await query_text(llm, system, user))


async def query_model(llm: LLM, model_type: type[ModelT], system: str, user: str) -> ModelT:
    data = await query_json(llm, system, user)
    return model_type.model_validate(data)
