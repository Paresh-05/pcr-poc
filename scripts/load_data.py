"""Load pcr_data 2.csv into the Postgres `cases` table (idempotent)."""
import csv
import json
import os
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "pcr_data 2.csv"
EXPECTED = ["dtc_code", "model_line", "customer_verbatim", "diagnosis_by_texpert", "failed_parts"]


def load_env(path: Path) -> None:
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def parse_parts(raw: str) -> list:
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(p) for p in val] if isinstance(val, list) else []


def main() -> None:
    load_env(ROOT / ".env")
    csv.field_size_limit(10**9)
    rows, skipped = [], 0
    with open(CSV_PATH, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EXPECTED:
            sys.exit(f"Unexpected columns: {reader.fieldnames}")
        for i, r in enumerate(reader, start=1):
            code = (r["dtc_code"] or "").strip().upper()
            model = (r["model_line"] or "").strip().upper()
            if not code or not model:
                skipped += 1
                continue
            rows.append((
                i, code, model,
                (r["customer_verbatim"] or "").strip(),
                (r["diagnosis_by_texpert"] or "").strip(),
                json.dumps(parse_parts(r["failed_parts"])),
            ))

    with psycopg.connect(
        host="127.0.0.1", port=int(os.environ["POSTGRES_PORT"]),
        user=os.environ["POSTGRES_USER"], password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    ) as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO cases (source_row, dtc_code, model_line, customer_verbatim, diagnosis, failed_parts)
               VALUES (%s, %s, %s, %s, %s, %s::jsonb)
               ON CONFLICT (source_row) DO NOTHING""",
            rows,
        )
        cur.execute("SELECT count(*) FROM cases")
        print(f"parsed={len(rows)} skipped={skipped} table_rows={cur.fetchone()[0]}")


if __name__ == "__main__":
    main()
