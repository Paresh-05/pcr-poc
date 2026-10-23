"""Ingest past-complaint (PCR) tickets from a CSV export into pcr_cases - content only, no embeddings
(summary + embedding is a separate later step, same as manuals).

Expected CSV columns (matches the real export format):
  ticket_no, updated_date, model_line, model_variant, fuel, customer_verbatim, comments, failed_parts, dtc_code

- comments: a multi-turn conversation thread ("Dealer: ...\\nTM: ..."); the last substantial turn is
  extracted as conclusion_text (the actual resolution), the rest is kept as comments_raw for citation only.
- failed_parts: a Python-list-literal string of "CODE-NAME" items; normalized to [{"part_code","part_name"}].

Usage: uv run python scripts/ingest_pcr.py path/to/pcr_export.csv
"""
import argparse
import ast
import asyncio
import csv
import logging
import re
import sys
from pathlib import Path

from psycopg.types.json import Json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import create_pool

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ingest_pcr")

_TURN_RE = re.compile(r"^\s*(Dealer|TM|CT)\s*:\s*", re.IGNORECASE | re.MULTILINE)
_MIN_CONCLUSION_LEN = 30


def parse_comments(raw: str) -> str:
    """Split the conversation into speaker turns; return the last substantial one (the real resolution)."""
    if not raw or not raw.strip():
        return ""
    splits = _TURN_RE.split(raw)
    # splits = ['', 'Dealer', ' text1...', 'TM', ' text2...', ...] when turns exist
    turns = []
    i = 1
    while i < len(splits) - 1:
        turns.append(splits[i + 1].strip())
        i += 2
    if not turns:
        return raw.strip()
    substantial = [t for t in turns if len(t) >= _MIN_CONCLUSION_LEN]
    return (substantial[-1] if substantial else turns[-1]).strip()


def parse_failed_parts(raw: str) -> list[dict]:
    if not raw or not raw.strip():
        return []
    try:
        items = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        items = [p.strip() for p in raw.strip("[]").replace('"', "").split(",") if p.strip()]
    out = []
    for item in items:
        item = str(item).strip()
        if "-" in item:
            code, name = item.split("-", 1)
            out.append({"part_code": code.strip(), "part_name": name.strip()})
        else:
            out.append({"part_code": "", "part_name": item})
    return out


def conclusion_matches_parts(conclusion_text: str, parts: list[dict]) -> bool | None:
    if not parts:
        return None
    text = conclusion_text.lower()
    return any(p["part_name"].split()[0].lower() in text for p in parts if p.get("part_name"))


async def main(csv_path: Path) -> None:
    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()

    inserted = skipped = flagged = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticket_no = (row.get("ticket_no") or "").strip()
            if not ticket_no:
                skipped += 1
                continue

            conclusion_text = parse_comments(row.get("comments") or "")
            failed_parts = parse_failed_parts(row.get("failed_parts") or "")
            match = conclusion_matches_parts(conclusion_text, failed_parts)
            if match is False:
                flagged += 1
                log.warning("ticket %s: conclusion text does not mention any logged failed_parts - review", ticket_no)

            customer_verbatim = (row.get("customer_verbatim") or "").strip()
            if not customer_verbatim and not conclusion_text:
                skipped += 1
                continue

            async with pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO pcr_cases (ticket_no, model_line, model_variant, fuel, dtc_code, "
                    "customer_verbatim, comments_raw, conclusion_text, failed_parts, conclusion_matches_parts, "
                    "updated_date) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (ticket_no) DO UPDATE SET conclusion_text = EXCLUDED.conclusion_text, "
                    "failed_parts = EXCLUDED.failed_parts",
                    (
                        ticket_no,
                        (row.get("model_line") or "").strip().upper(),
                        (row.get("model_variant") or "").strip() or None,
                        (row.get("fuel") or "").strip() or None,
                        (row.get("dtc_code") or "").strip().upper(),
                        customer_verbatim,
                        row.get("comments") or "",
                        conclusion_text,
                        Json(failed_parts),
                        match,
                        row.get("updated_date") or None,
                    ),
                )
            inserted += 1
            if inserted % 50 == 0:
                log.info("ingested %d tickets so far...", inserted)

    await pool.close()
    log.info("done: %d inserted, %d skipped, %d flagged for conclusion/part mismatch", inserted, skipped, flagged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    # Windows needs a selector event loop for psycopg async (see run.py).
    asyncio.run(main(args.csv_path), loop_factory=asyncio.SelectorEventLoop)
