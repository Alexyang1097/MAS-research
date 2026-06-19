from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import APIError, AsyncOpenAI, BadRequestError

from .llm_utils import extract_json_object


@dataclass(frozen=True)
class JudgeBackendConfig:
    api_key: str
    base_url: str
    model: str
    fallback_models: list[str] = field(default_factory=list)
    temperature: float = 0.0


class OpenAICompatibleJudgeBackend:
    """Small OpenAI-compatible backend used only for rubric judging.

    This intentionally does not use model_library, so newly released judge models can be tried
    by raw API model name even before they appear in the model registry.
    """

    def __init__(self, config: JudgeBackendConfig):
        self._config = config
        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    async def judge_json(self, *, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for model in self._candidate_models():
            try:
                result = await self._complete_json(system=system, payload=payload, model=model, use_json_mode=True)
                result["_judge_model_used"] = model
                return result
            except (BadRequestError, APIError) as error:
                last_error = error
                if not _looks_like_json_mode_error(error):
                    continue
                try:
                    result = await self._complete_json(system=system, payload=payload, model=model, use_json_mode=False)
                    result["_judge_model_used"] = model
                    return result
                except (BadRequestError, APIError) as retry_error:
                    last_error = retry_error
                    continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("No judge model candidates were provided.")

    async def _complete_json(
        self,
        *,
        system: str,
        payload: dict[str, Any],
        model: str,
        use_json_mode: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": self._config.temperature,
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or "{}"
        return extract_json_object(content)

    def _candidate_models(self) -> list[str]:
        candidates = [self._config.model, *self._config.fallback_models]
        deduped: list[str] = []
        for model in candidates:
            if model and model not in deduped:
                deduped.append(model)
        return deduped


def _looks_like_json_mode_error(error: Exception) -> bool:
    text = str(error).lower()
    return "response_format" in text or "json" in text
