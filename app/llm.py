"""OpenAI access: plain-text summarization (for the case digest) and the structured chat turn."""
import json
import re
from typing import Literal, Protocol

from openai import AsyncOpenAI, BadRequestError, LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from app.config import Settings


class Cause(BaseModel):
    cause: str
    confidence: Literal["high", "medium", "low"]
    support: str  # where it comes from, e.g. "manual: 18.3 Engine Control Module" or "past cases"


class ChatTurn(BaseModel):
    reply: str
    state: Literal["asking", "concluded"]
    likely_causes: list[Cause]
    next_checks: list[str]


class LLM(Protocol):
    async def summarize(self, system: str, user: str) -> str: ...

    async def converse(self, system: str, messages: list[dict]) -> ChatTurn: ...


def _extract_json(text: str) -> ChatTurn:
    """Parse the first {...} block; some models wrap JSON in prose or code fences."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in LLM reply")
    return ChatTurn.model_validate_json(match.group(0))


class OpenAILLM:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key.get_secret_value(), base_url=settings.llm_base_url, timeout=90, max_retries=2
        )

    async def summarize(self, system: str, user: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()

    async def converse(self, system: str, messages: list[dict]) -> ChatTurn:
        full = [{"role": "system", "content": system}, *messages]
        try:
            completion = await self.client.chat.completions.parse(
                model=self.model, messages=full, response_format=ChatTurn
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except (BadRequestError, LengthFinishReasonError, ValidationError):
            pass  # model without structured-output support: fall back to prompted JSON

        schema = json.dumps(ChatTurn.model_json_schema())
        full[0] = {
            "role": "system",
            "content": system + "\n\nReply with ONLY one JSON object matching this JSON Schema, no other text:\n" + schema,
        }
        completion = await self.client.chat.completions.create(model=self.model, messages=full)
        return _extract_json(completion.choices[0].message.content or "")
