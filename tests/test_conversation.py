"""Conversation flow against the real database with a fake LLM, using a throwaway DTC that is cleaned up after."""
import asyncio
import uuid

import pytest

from app import conversations, digest
from app.config import Settings
from app.db import create_pool
from app.llm import ChatTurn, Cause
from app.manual import Section, fit_budget

DTC, MODEL = "ZZTEST1", "TESTMODEL"


class FakeLLM:
    def __init__(self):
        self.summaries = 0
        self.systems: list[str] = []
        self.histories: list[list[dict]] = []

    async def summarize(self, system: str, user: str) -> str:
        self.summaries += 1
        return f"- ISSUE: sample issue COUNT: 1 (call {self.summaries})"

    async def converse(self, system: str, messages: list[dict]) -> ChatTurn:
        self.systems.append(system)
        self.histories.append(messages)
        return ChatTurn(
            reply=f"reply {len(self.histories)}", state="asking",
            likely_causes=[Cause(cause="c", confidence="low", support="past cases")], next_checks=["check A"],
        )


def test_fit_budget_keeps_total_within_limit():
    sections = [Section(f"m{i}", "x" * 20_000) for i in range(21)]
    trimmed, truncated = fit_budget(sections)
    assert truncated and sum(len(s.text) for s in trimmed) <= 21 * 2_900


async def _seed(pool, n_cases: int) -> None:
    async with pool.connection() as conn:
        for i in range(n_cases):
            await conn.execute(
                "INSERT INTO cases (source_row, dtc_code, model_line, customer_verbatim, diagnosis, failed_parts) "
                "VALUES (%s,%s,%s,%s,%s,%s::jsonb)",
                (-1000 - i, DTC, MODEL, "engine light on", f"replaced sensor {i}", '["SENSOR-1"]'),
            )
        await conn.execute(
            "INSERT INTO manual_sections (dtc_code, model_line, module_name, content) VALUES (%s,%s,'ECM TEST','Check connector X1 first.')",
            (DTC, MODEL),
        )


async def _cleanup(pool) -> None:
    async with pool.connection() as conn:
        await conn.execute("DELETE FROM conversations WHERE dtc_code = %s", (DTC,))
        await conn.execute("DELETE FROM case_digests WHERE dtc_code = %s", (DTC,))
        await conn.execute("DELETE FROM manual_sections WHERE dtc_code = %s", (DTC,))
        await conn.execute("DELETE FROM cases WHERE dtc_code = %s", (DTC,))


async def _scenario() -> None:
    pool = create_pool(Settings())
    await pool.open()
    try:
        await _cleanup(pool)
        await _seed(pool, digest.DIRECT_LIMIT + digest.BATCH_SIZE + 5)  # forces batching + merge
        llm = FakeLLM()

        started = await conversations.start_conversation(pool, llm, DTC, MODEL)
        cid = uuid.UUID(started["conversation_id"])
        assert started["evidence"]["total_cases"] == digest.DIRECT_LIMIT + digest.BATCH_SIZE + 5
        assert started["evidence"]["manual"] == {"status": "used", "modules": ["ECM TEST"], "truncated": False}
        assert llm.summaries >= 3  # batches + merge
        assert "Check connector X1 first." in llm.systems[0] and "engine light on" in llm.systems[0]

        first_summaries = llm.summaries
        await conversations.start_conversation(pool, llm, DTC, MODEL)
        assert llm.summaries == first_summaries  # digest served from cache

        reply = await conversations.send_message(pool, llm, cid, "connector looks fine")
        assert reply["turn"]["reply"].startswith("reply")
        assert llm.histories[-1][-1] == {"role": "user", "content": "connector looks fine"}
        assert [m["role"] for m in llm.histories[-1]] == ["user", "assistant", "user"]

        shown = await conversations.get_conversation(pool, cid)
        assert [m["role"] for m in shown["messages"]] == ["assistant", "user", "assistant"]  # opening prompt hidden

        with pytest.raises(conversations.NotFound):
            await conversations.send_message(pool, llm, uuid.uuid4(), "hi")

        async with pool.connection() as conn:
            for _ in range(conversations.MAX_MESSAGES):
                await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES (%s,'user','x')", (cid,))
        with pytest.raises(conversations.LimitReached):
            await conversations.send_message(pool, llm, cid, "one more")

        none = await conversations.start_conversation(pool, llm, "ZZTEST9", MODEL)  # no cases, no manual
        assert none["evidence"]["scope"] == "none" and none["evidence"]["manual"]["status"] == "not_found"
        assert "not available" in llm.systems[-1]
    finally:
        await _cleanup(pool)
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM conversations WHERE dtc_code = 'ZZTEST9'")
        await pool.close()


def test_conversation_flow():
    asyncio.run(_scenario(), loop_factory=asyncio.SelectorEventLoop)
