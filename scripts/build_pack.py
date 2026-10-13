"""Build (or rebuild) the knowledge pack for one fault code and print a summary.

Usage: uv run --system-certs python scripts/build_pack.py P187973 MG_COMET_EV
Makes ONE paid AI call (model: LLM_PACK_MODEL, or LLM_MODEL). The cost is recorded in llm_calls / run_costs.
"""
import asyncio
import sys
import uuid

from app.config import get_settings
from app.db import create_pool
from app.llm import OpenAILLM
from app.pack import get_or_build_pack
from app.usage import RunInfo, UsageRecorder, current_run


async def main(dtc: str, model: str) -> None:
    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()
    try:
        llm = OpenAILLM(settings, UsageRecorder(pool))
        run_id = uuid.uuid4()
        current_run.set(RunInfo(run_id, "pack", dtc, model))
        pack = await get_or_build_pack(pool, llm, dtc, model, force=True)
        print(f"checks: {pack['checks']} | stats: {pack['stats']['distinct_cases']} distinct of {pack['stats']['raw_cases']} cases")
        for c in pack["causes"]:
            print(f"- {c['cause']} [{c['origin']}] {c['share_pct']}% ({c['cases_counted']} cases) parts={c['matching_parts']} quotes={len(c['quotes'])}")
        print(f"questions: {len(pack['question_bank'])} | gaps: {pack['gaps']}")
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT calls, input_tokens, output_tokens, reasoning_tokens, cost_usd FROM run_costs WHERE run_id = %s", (run_id,))
            print("cost:", await cur.fetchone())
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1].upper(), sys.argv[2].upper()), loop_factory=asyncio.SelectorEventLoop)
