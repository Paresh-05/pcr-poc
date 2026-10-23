"""OpenAI access: embeddings, schema-checked structured output."""
import json
import logging
import re
import time
from typing import Protocol, TypeVar

from openai import AsyncOpenAI, BadRequestError, LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.usage import UsageRecorder, usage_from_response

logger = logging.getLogger("dtcbot.llm")

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    async def embed(self, text: str) -> list[float]: ...

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
        self.embedding_model = settings.embedding_model
        self.effort = settings.llm_reasoning_effort
        self.recorder = recorder
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key.get_secret_value(), base_url=settings.llm_base_url, timeout=300, max_retries=2
        )

    def _extra(self, purpose: str, effort_override: str | None) -> dict:
        effort = effort_override if effort_override is not None else self.effort
        return {"reasoning_effort": effort} if effort and purpose != "embed" else {}

    async def _call(self, fn, purpose: str, effort_override: str | None, **kwargs):
        try:
            return await fn(**kwargs, **self._extra(purpose, effort_override))
        except BadRequestError as exc:
            if "reasoning_effort" in str(exc) and (effort_override or self.effort):
                self.effort = None
                return await fn(**kwargs)
            raise

    async def _record(self, resp, purpose: str, model: str, started: float) -> None:
        if self.recorder:
            await self.recorder.record(model, purpose, usage_from_response(resp), int((time.monotonic() - started) * 1000))

    async def embed(self, text: str) -> list[float]:
        started = time.monotonic()
        resp = await self.client.embeddings.create(model=self.embedding_model, input=text[:8000])
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info("embed call: model=%s chars=%d latency=%dms", self.embedding_model, len(text), elapsed_ms)
        await self._record(resp, "embed", self.embedding_model, started)
        return resp.data[0].embedding

    async def structured(self, system: str, user: str, schema: type[T], purpose: str, effort: str | None = None) -> T:
        """`effort` overrides the configured default reasoning effort for this one call (e.g. "high" for a
        call that benefits from deeper reasoning, like turning a messy manual into structured analysis)."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logger.info("structured call: purpose=%s model=%s schema=%s", purpose, self.model, schema.__name__)
        started = time.monotonic()
        try:
            completion = await self._call(
                self.client.chat.completions.parse, purpose, effort, model=self.model, messages=messages, response_format=schema
            )
            elapsed_ms = int((time.monotonic() - started) * 1000)
            usage = usage_from_response(completion)
            logger.info(
                "structured call OK (native): purpose=%s model=%s latency=%dms input_tokens=%s output_tokens=%s",
                purpose, self.model, elapsed_ms, usage.input_tokens if usage else None, usage.output_tokens if usage else None,
            )
            await self._record(completion, purpose, self.model, started)
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except (BadRequestError, LengthFinishReasonError, ValidationError) as exc:
            logger.warning("structured call: native structured output failed (%s), falling back to prompted JSON", type(exc).__name__)

        messages[0] = {
            "role": "system",
            "content": system + "\n\nReply with ONLY one JSON object matching this JSON Schema, no other text:\n"
            + json.dumps(schema.model_json_schema()),
        }
        started = time.monotonic()
        completion = await self._call(self.client.chat.completions.create, purpose, effort, model=self.model, messages=messages)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        usage = usage_from_response(completion)
        logger.info(
            "structured call OK (prompted-JSON fallback): purpose=%s model=%s latency=%dms input_tokens=%s output_tokens=%s",
            purpose, self.model, elapsed_ms, usage.input_tokens if usage else None, usage.output_tokens if usage else None,
        )
        await self._record(completion, purpose, self.model, started)
        return _extract_json(completion.choices[0].message.content or "", schema)
