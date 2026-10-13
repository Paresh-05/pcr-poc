from decimal import Decimal
from types import SimpleNamespace

from app.usage import Price, Usage, compute_cost, usage_from_response

LUNA = Price(Decimal("0.10"), Decimal("0.01"), Decimal("0.50"))


def test_cost_splits_cached_and_uncached_input_and_bills_output():
    # 100k input of which 60k cached; 10k output (4k of it reasoning, already inside output)
    cost = compute_cost(LUNA, Usage(input_tokens=100_000, cached_input_tokens=60_000, output_tokens=10_000, reasoning_tokens=4_000))
    assert cost == Decimal("0.004") + Decimal("0.0006") + Decimal("0.005")  # 40k*0.10 + 60k*0.01 + 10k*0.50 per million


def test_cost_is_none_without_a_price_and_never_negative_input():
    assert compute_cost(None, Usage(10, 10)) is None
    assert compute_cost(LUNA, Usage(input_tokens=5, cached_input_tokens=9, output_tokens=0)) == Decimal("9") * Decimal("0.01") / 1_000_000


def test_usage_is_read_from_an_openai_style_response():
    resp = SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=1200, completion_tokens=300, total_tokens=1500,
        prompt_tokens_details=SimpleNamespace(cached_tokens=1000),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=120),
    ))
    u = usage_from_response(resp)
    assert (u.input_tokens, u.cached_input_tokens, u.output_tokens, u.reasoning_tokens, u.total_tokens) == (1200, 1000, 300, 120, 1500)
    assert usage_from_response(SimpleNamespace(usage=None)) is None
    bare = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, prompt_tokens_details=None, completion_tokens_details=None))
    assert usage_from_response(bare).cached_input_tokens == 0


def test_reasoning_effort_is_sent_per_purpose_and_dropped_if_unsupported():
    import asyncio

    from openai import BadRequestError

    from app.llm import OpenAILLM

    llm = OpenAILLM.__new__(OpenAILLM)  # no network client needed for this check
    llm.step_effort, llm.effort = "none", "medium"
    assert llm._extra("step") == {"reasoning_effort": "none"} and llm._extra("propose") == {"reasoning_effort": "none"}
    assert llm._extra("conclude") == {"reasoning_effort": "medium"} and llm._extra("digest_batch") == {"reasoning_effort": "medium"}

    seen = []

    async def fake_api(**kw):
        seen.append(kw)
        if "reasoning_effort" in kw:
            import httpx
            raise BadRequestError("unsupported reasoning_effort", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)
        return "ok"

    assert asyncio.run(llm._call(fake_api, "step", model="m")) == "ok"
    assert "reasoning_effort" in seen[0] and "reasoning_effort" not in seen[1] and llm._extra("step") == {}
