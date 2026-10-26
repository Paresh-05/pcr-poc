"""Samples resolved pcr_cases into a small golden set for scripts/eval_diagnosis.py: real
(symptom -> actual cause/parts) pairs, grouped by (dtc_code, model_line) so common codes aren't
over-represented. Writes scripts/golden_set.json.

Caveat: each sampled case is itself part of the retrieval corpus, so this is a regression/sanity
check (did a prompt or retrieval change make known-good cases worse), not a true held-out eval -
the exact case text will always be retrievable by itself.

Usage: uv run python scripts/build_golden_set.py [--per-code N] [--max-codes N]
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import get_settings
from app.db import create_pool

OUT_PATH = Path(__file__).resolve().parent / "golden_set.json"
SEED = 42


async def main(per_code: int, max_codes: int | None) -> None:
    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()

    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT ticket_no, dtc_code, model_line, customer_verbatim, conclusion_text, failed_parts "
            "FROM pcr_cases WHERE resolution_status = 'resolved' "
            "AND customer_verbatim IS NOT NULL AND conclusion_text IS NOT NULL "
            "AND failed_parts IS NOT NULL AND failed_parts != '[]'::jsonb"
        )
        cases = await cur.fetchall()
    await pool.close()

    by_code: dict[tuple[str, str], list[dict]] = {}
    for c in cases:
        by_code.setdefault((c["dtc_code"], c["model_line"]), []).append(c)

    groups = sorted(by_code.items(), key=lambda kv: len(kv[1]), reverse=True)
    if max_codes:
        groups = groups[:max_codes]

    rng = random.Random(SEED)
    golden = []
    for (dtc_code, model_line), group in groups:
        picked = rng.sample(group, min(per_code, len(group)))
        for c in picked:
            parts = [p.get("part_name") or p.get("part_code", "") for p in (c["failed_parts"] or [])]
            golden.append({
                "ticket_no": c["ticket_no"],
                "dtc_code": dtc_code,
                "model_line": model_line,
                "query_text": c["customer_verbatim"],
                "expected_cause": c["conclusion_text"],
                "expected_parts": [p for p in parts if p],
            })

    OUT_PATH.write_text(json.dumps(golden, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(golden)} golden case(s) across {len(groups)} (dtc_code, model_line) group(s) -> {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-code", type=int, default=2, help="max cases sampled per (dtc_code, model_line)")
    parser.add_argument("--max-codes", type=int, default=20, help="max number of (dtc_code, model_line) groups")
    args = parser.parse_args()
    asyncio.run(main(args.per_code, args.max_codes), loop_factory=asyncio.SelectorEventLoop)
