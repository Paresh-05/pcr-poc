"""OpenAI access: plain-text summarization (case digest) and schema-checked structured output."""
import json
import re
import time
from typing import Protocol, TypeVar

from openai import AsyncOpenAI, BadRequestError, LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.usage import UsageRecorder, usage_from_response

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    async def summarize(self, system: str, user: str, purpose: str = "summarize") -> str: ...

    async def structured(self, system: str, user: str, schema: type[T], purpose: str) -> T: ...


def _extract_json(text: str, schema: type[T]) -> T:
    """Parse the first {...} block; some models wrap JSON in prose or code fences."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in LLM reply")
    return schema.model_validate_json(match.group(0))


class OpenAILLM:
    def __init__(self, settings: Settings, recorder: UsageRecorder | None = None) -> None:
        self.model = settings.llm_model
        self.effort = settings.llm_reasoning_effort
        self.step_effort = settings.llm_step_effort
        self.recorder = recorder
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key.get_secret_value(), base_url=settings.llm_base_url, timeout=90, max_retries=2
        )

    def _extra(self, purpose: str) -> dict:
        effort = self.step_effort if purpose in ("propose", "step") else self.effort
        return {"reasoning_effort": effort} if effort else {}

    async def _call(self, fn, purpose: str, **kwargs):
        """Call the API with the configured reasoning effort; if the model does not support it, stop sending it."""
        try:
            return await fn(**kwargs, **self._extra(purpose))
        except BadRequestError as exc:
            if "reasoning_effort" in str(exc) and (self.effort or self.step_effort):
                self.effort = self.step_effort = None
                return await fn(**kwargs)
            raise

    async def _record(self, resp, purpose: str, started: float) -> None:
        if self.recorder:
            await self.recorder.record(self.model, purpose, usage_from_response(resp), int((time.monotonic() - started) * 1000))

    async def summarize(self, system: str, user: str, purpose: str = "summarize") -> str:
        started = time.monotonic()
        resp = await self._call(
            self.client.chat.completions.create, purpose,
            model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        await self._record(resp, purpose, started)
        return (resp.choices[0].message.content or "").strip()

    async def structured(self, system: str, user: str, schema: type[T], purpose: str) -> T:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        started = time.monotonic()
        try:
            completion = await self._call(
                self.client.chat.completions.parse, purpose, model=self.model, messages=messages, response_format=schema
            )
            await self._record(completion, purpose, started)
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except (BadRequestError, LengthFinishReasonError, ValidationError):
            pass  # model without structured-output support: fall back to prompted JSON

        messages[0] = {
            "role": "system",
            "content": system + "\n\nReply with ONLY one JSON object matching this JSON Schema, no other text:\n"
            + json.dumps(schema.model_json_schema()),
        }
        started = time.monotonic()
        completion = await self._call(self.client.chat.completions.create, purpose, model=self.model, messages=messages)
        await self._record(completion, purpose, started)
        return _extract_json(completion.choices[0].message.content or "", schema)
