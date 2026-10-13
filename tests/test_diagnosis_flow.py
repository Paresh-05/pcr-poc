"""Full diagnosis against the real database with a fake LLM that reports token usage; cleans up after itself."""
import asyncio
import uuid
from decimal import Decimal

import pytest

from app import diagnosis
from app.config import Settings
from app.db import create_pool
from app.diagnosis_graph import Candidate, Conclusion, StepOut, build_graph
from app.pack import Pack, PackCause, PackQuestion, Quote
from app.usage import UsageRecorder, Usage

DTC, MODEL, NO_DATA_DTC, PRICED = "ZZTEST1", "TESTMODEL", "ZZTEST9", "test-model"
TESTMODEL = MODEL


class FakeLLM:
    """Scripted answers; every call reports fixed token usage through the real UsageRecorder."""

    def __init__(self, recorder: UsageRecorder):
        self.recorder = recorder
        self.n_calls = 0
        self.n_steps = 0

    async def _bill(self, purpose: str, usage: Usage) -> None:
        self.n_calls += 1
        await self.recorder.record(PRICED, purpose, usage, latency_ms=7)

    async def summarize(self, system, user, purpose="summarize"):
        await self._bill(purpose, Usage(input_tokens=1000, output_tokens=100))
        return f"- ISSUE x COUNT: 1 ({purpose})"

    async def structured(self, system, user, schema, purpose):
        await self._bill(purpose, Usage(input_tokens=2000, cached_input_tokens=1500, output_tokens=300, reasoning_tokens=50))
        if schema is Pack:  # the once-per-code knowledge pack
            return Pack(
                fault_meaning="m", setting_conditions="s", gaps=[],
                causes=[
                    PackCause(cause="Sensor X", origin="both", matching_parts=["SENSOR-1"], replaces_nothing=False,
                              quotes=[Quote(source="manual", text="Is the connector damaged?")], how_to_tell=["t"], repair_steps=["s1"]),
                    PackCause(cause="Wiring", origin="manual", matching_parts=[], replaces_nothing=False,
                              quotes=[Quote(source="manual", text="Repair the signal circuit")], how_to_tell=[], repair_steps=[]),
                    PackCause(cause="ECU", origin="manual", matching_parts=[], replaces_nothing=False,
                              quotes=[Quote(source="manual", text="this quote is invented")], how_to_tell=[], repair_steps=[]),
                ],
                question_bank=[PackQuestion(question="Is check number 1 positive?", source="manual",
                                            yes_points_to=["Sensor X"], no_points_to=["Wiring"])],
            )
        if schema is Conclusion:
            return Conclusion(problem="Sensor X", confidence="high", evidence=["e"], how_we_got_here=["h"], ruled_out=[],
                              uncertain=["u"], fix_steps=["s1"], parts=["p"], verify_fix=["v"], safety_note="n", alternatives=[])
        script = [[("Sensor X", 55), ("Wiring", 30)], [("Sensor X", 70), ("Wiring", 20)],
                  [("Sensor X", 80), ("Wiring", 15)], [("Sensor X", 90), ("Wiring", 6)]]
        likes = script[self.n_steps]
        self.n_steps += 1
        return StepOut(candidates=[Candidate(cause=c, likelihood=n, reasoning="r") for c, n in likes],
                       what_it_showed=f"answer {self.n_steps} moved the ranking", next_question=f"Is check number {self.n_steps + 1} positive?", ready_to_conclude=False)


async def _seed(pool) -> None:
    async with pool.connection() as conn:
        for i in range(60):
            await conn.execute(
                "INSERT INTO cases (source_row, dtc_code, model_line, customer_verbatim, diagnosis, failed_parts) "
                "VALUES (%s,%s,%s,'engine light on',%s,'[\"SENSOR-1\"]'::jsonb)", (-2000 - i, DTC, MODEL, f"replaced sensor {i}"))
        await conn.execute("INSERT INTO manual_model_map (manual_model, model_line) VALUES ('TESTMANUAL', %s)", (MODEL,))
        cur = await conn.execute("INSERT INTO manual_dtcs (manual_model, dtc_code) VALUES ('TESTMANUAL', %s) RETURNING id", (DTC,))
        (dtc_id,) = await cur.fetchone()
        await conn.execute(
            "INSERT INTO manual_subsystems (manual_dtc_id, position, subsystem, content) VALUES (%s,1,'ECM TEST',%s)",
            (dtc_id, f"DTC {DTC} Diagnostic Description. Circuit/System Inspection. "
                     "1. Is the connector damaged? Yes. Repair the connector. No. Proceed to the next inspection. "
                     "2. Is the signal correct? Yes. Diagnosis completed. No. Repair the signal circuit. Diagnosis completed."),
        )
        await conn.execute(
            "INSERT INTO model_prices (model, input_per_mtok, cached_input_per_mtok, output_per_mtok, source) "
            "VALUES (%s, 0.10, 0.01, 0.50, 'test') ON CONFLICT (model) DO NOTHING", (PRICED,))


async def _cleanup(pool) -> None:
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT id FROM diagnoses WHERE dtc_code = ANY(%s)", ([DTC, NO_DATA_DTC],))
        ids = [r[0] for r in await cur.fetchall()]
        await conn.execute("DELETE FROM llm_calls WHERE model = %s OR run_id = ANY(%s)", (PRICED, ids))
        await conn.execute("DELETE FROM run_costs WHERE run_id = ANY(%s)", (ids,))
        await conn.execute("DELETE FROM diagnoses WHERE id = ANY(%s)", (ids,))
        await conn.execute("DELETE FROM knowledge_packs WHERE dtc_code = %s", (DTC,))
        await conn.execute("DELETE FROM manual_dtcs WHERE manual_model = 'TESTMANUAL'")
        await conn.execute("DELETE FROM manual_model_map WHERE manual_model = 'TESTMANUAL'")
        await conn.execute("DELETE FROM cases WHERE dtc_code = %s", (DTC,))
        await conn.execute("DELETE FROM model_prices WHERE model = %s", (PRICED,))


async def _scenario() -> None:
    pool = create_pool(Settings())
    await pool.open()
    try:
        await _cleanup(pool)
        await _seed(pool)
        llm = FakeLLM(UsageRecorder(pool))
        graph = build_graph(llm)

        started = await diagnosis.start_diagnosis(pool, llm, graph, DTC, MODEL)
        did = uuid.UUID(started["diagnosis_id"])
        assert started["status"] == "asking" and started["question"] == "Is check number 1 positive?"
        assert started["answer_options"] == ["yes", "no", "not_sure"]
        assert started["evidence"]["manual"]["status"] == "used" and started["evidence"]["total_cases"] == 60
        assert [c["cause"] for c in started["candidates"]] == ["Sensor X", "Wiring"]  # the invented-quote cause was dropped
        assert started["start_candidates"] == started["candidates"]
        assert started["cost"]["calls"] == 1  # only the knowledge-pack call; the opening needs no AI
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT pack FROM knowledge_packs WHERE dtc_code = %s", (DTC,))
            (stored,) = await cur.fetchone()
        assert stored["causes"][0]["cases_counted"] == 60 and stored["causes"][0]["share_pct"] == 100
        assert stored["checks"]["quotes_dropped"] == 1 and stored["stats"]["distinct_cases"] == 60

        step = await diagnosis.answer_diagnosis(pool, llm, graph, did, "yes", "rust visible")
        assert step["status"] == "asking" and step["questions_asked"] == 1 and step["candidates"][0]["likelihood"] == 65
        assert step["explanation"] == "answer 1 moved the ranking"
        step = await diagnosis.answer_diagnosis(pool, llm, graph, did, "not_sure", "")
        assert step["status"] == "asking" and step["questions_asked"] == 2
        step = await diagnosis.answer_diagnosis(pool, llm, graph, did, "no", "")
        assert step["status"] == "asking"  # only two Yes/No answers so far: "not sure" does not count
        done = await diagnosis.answer_diagnosis(pool, llm, graph, did, "yes", "")
        assert done["status"] == "concluded" and done["conclusion"]["problem"] == "Sensor X" and done["question"] is None
        assert done["conclusion"]["confidence"] == "high" and "confidence_note" in done["conclusion"]
        assert done["cost"]["calls"] == 6  # pack + 4 answer steps + conclusion

        # cost tables: 6 structured calls, priced at 0.10 / 0.01 / 0.50 per million tokens
        expected = Decimal("6") * Decimal("0.000215")
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT calls, input_tokens, cached_input_tokens, output_tokens, reasoning_tokens, cost_usd, unpriced_calls FROM run_costs WHERE run_id = %s", (did,))
            calls, i, c, o, r, cost, unpriced = await cur.fetchone()
            cur = await conn.execute("SELECT count(*), sum(cost_usd), sum(total_tokens) FROM llm_calls WHERE run_id = %s", (did,))
            n_rows, summed, total_tokens = await cur.fetchone()
        assert (calls, unpriced) == (6, 0) and n_rows == 6
        assert (i, c, o, r) == (6 * 2000, 6 * 1500, 6 * 300, 6 * 50)
        assert cost == summed == expected and total_tokens == i + o
        assert (await diagnosis.list_calls(pool, did))["run_total"]["cost_usd"] == pytest.approx(float(expected))

        shown = await diagnosis.get_diagnosis(pool, did)
        assert shown["status"] == "concluded" and [s["answer"] for s in shown["steps"]] == ["yes", "not_sure", "no", "yes"]
        assert shown["steps"][0]["note"] == "rust visible" and shown["steps"][3]["candidates_after"][0]["likelihood"] == 91
        assert shown["steps"][0]["explanation"] == "answer 1 moved the ranking" and shown["start_candidates"]

        # outcome feedback: stored once per diagnosis, private details removed, later answers replace earlier ones
        saved = await diagnosis.save_outcome(pool, did, "worked", "Sensor X failed, call 9876543210", ["SENSOR-1"], "all good")
        assert saved["result"] == "worked" and "9876543210" not in saved["actual_cause"] and saved["parts_replaced"] == ["SENSOR-1"]
        saved = await diagnosis.save_outcome(pool, did, "partly", "", [], "")
        assert saved["result"] == "partly" and saved["parts_replaced"] == [] and saved["actual_cause"] == ""
        assert (await diagnosis.get_diagnosis(pool, did))["outcome"]["result"] == "partly"
        summary = await diagnosis.outcome_summary(pool)
        assert summary["with_feedback"] >= 1 and summary["partly"] >= 1 and TESTMODEL in summary["by_model"]

        with pytest.raises(diagnosis.NotAsking):
            await diagnosis.answer_diagnosis(pool, llm, graph, did, "no", "")
        with pytest.raises(diagnosis.NotFound):
            await diagnosis.answer_diagnosis(pool, llm, graph, uuid.uuid4(), "yes", "")

        before = llm.n_calls
        empty = await diagnosis.start_diagnosis(pool, llm, graph, NO_DATA_DTC, MODEL)  # no cases, no manual
        assert empty["status"] == "no_evidence" and empty["question"] is None and llm.n_calls == before and empty["cost"] is None
        with pytest.raises(diagnosis.NotConcluded):  # no feedback before there is a conclusion
            await diagnosis.save_outcome(pool, uuid.UUID(empty["diagnosis_id"]), "worked", "", [], "")
        with pytest.raises(diagnosis.NotFound):
            await diagnosis.save_outcome(pool, uuid.uuid4(), "worked", "", [], "")
    finally:
        await _cleanup(pool)
        await pool.close()


def test_full_diagnosis_flow_and_cost_tracking():
    asyncio.run(_scenario(), loop_factory=asyncio.SelectorEventLoop)
