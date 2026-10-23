"""Token usage and cost tracking: one llm_calls row per model call, one run_costs row per run."""
import contextvars
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal

from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger("dtcbot.usage")

_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class RunInfo:
    run_id: uuid.UUID
    run_type: str
    dtc_code: str | None = None
    model_line: str | None = None


# Every model call made while this is set is attributed to the run.
current_run: contextvars.ContextVar[RunInfo | None] = contextvars.ContextVar("current_run", default=None)


@dataclass(frozen=True)
class Usage:
    input_tokens: int  # all prompt tokens, including cached ones
    output_tokens: int  # all completion tokens, including reasoning ones
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Price:
    input_per_mtok: Decimal
    cached_input_per_mtok: Decimal
    output_per_mtok: Decimal


def compute_cost(price: Price | None, usage: Usage) -> Decimal | None:
    """Cost in USD. Uncached input at the input rate, cached input at the cached rate, all output (incl. reasoning)."""
    if price is None:
        return None
    uncached = max(usage.input_tokens - usage.cached_input_tokens, 0)
    return (
        uncached * price.input_per_mtok
        + usage.cached_input_tokens * price.cached_input_per_mtok
        + usage.output_tokens * price.output_per_mtok
    ) / _MILLION


def usage_from_response(resp) -> Usage | None:
    """Read token counts from an OpenAI-style response; None if the provider sent no usage."""
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
    # Embeddings responses have no completion_tokens (no output text); treat as 0.
    return Usage(
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
    )


class UsageRecorder:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def record(self, model: str, purpose: str, usage: Usage | None, latency_ms: int) -> None:
        """Never raises: cost tracking must not break a diagnosis."""
        if usage is None:
            return
        try:
            run = current_run.get()
            async with self.pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT input_per_mtok, cached_input_per_mtok, output_per_mtok FROM model_prices WHERE model = %s",
                    (model,),
                )
                row = await cur.fetchone()
                cost = compute_cost(Price(*row) if row else None, usage)
                await conn.execute(
                    "INSERT INTO llm_calls (run_id, run_type, purpose, model, input_tokens, cached_input_tokens, "
                    "output_tokens, reasoning_tokens, total_tokens, latency_ms, cost_usd) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (run.run_id if run else None, run.run_type if run else "none", purpose, model,
                     usage.input_tokens, usage.cached_input_tokens, usage.output_tokens, usage.reasoning_tokens,
                     usage.total_tokens, latency_ms, cost),
                )
                if run:
                    await conn.execute(
                        "INSERT INTO run_costs (run_id, run_type, dtc_code, model_line, calls, unpriced_calls, "
                        "input_tokens, cached_input_tokens, output_tokens, reasoning_tokens, cost_usd) "
                        "VALUES (%s,%s,%s,%s,1,%s,%s,%s,%s,%s,%s) "
                        "ON CONFLICT (run_id) DO UPDATE SET calls = run_costs.calls + 1, "
                        "unpriced_calls = run_costs.unpriced_calls + EXCLUDED.unpriced_calls, "
                        "input_tokens = run_costs.input_tokens + EXCLUDED.input_tokens, "
                        "cached_input_tokens = run_costs.cached_input_tokens + EXCLUDED.cached_input_tokens, "
                        "output_tokens = run_costs.output_tokens + EXCLUDED.output_tokens, "
                        "reasoning_tokens = run_costs.reasoning_tokens + EXCLUDED.reasoning_tokens, "
                        "cost_usd = run_costs.cost_usd + EXCLUDED.cost_usd, updated_at = now()",
                        (run.run_id, run.run_type, run.dtc_code, run.model_line, 0 if cost is not None else 1,
                         usage.input_tokens, usage.cached_input_tokens, usage.output_tokens, usage.reasoning_tokens,
                         cost if cost is not None else Decimal(0)),
                    )
            logger.info("usage recorded: model=%s purpose=%s tokens=%d cost_usd=%s latency=%dms",
                        model, purpose, usage.total_tokens, cost, latency_ms)
        except Exception as exc:  # noqa: BLE001
            logger.error("usage recording failed: %s", type(exc).__name__)
