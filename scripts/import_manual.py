"""Load manual content from a JSON file into the manual_sections table (idempotent upsert).

Usage: uv run --system-certs python scripts/import_manual.py data/manual_sections.json

File format: a JSON list; each item is one module of one DTC for one model:
  {"dtc_code": "P100016", "model_line": "HECTOR", "module_name": "18.3 Engine Control Module 1.5T",
   "content": "plain text of the manual section"}
`content` is plain text (no HTML). model_line must match the names used in the cases table (e.g. HECTOR, MG_ZS_EV).
"""
import json
import os
import re
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
DTC = re.compile(r"^[A-Z][0-9A-Z]{5,7}$")
MODEL = re.compile(r"^[A-Z0-9_]{2,32}$")
MAX_CONTENT = 100_000
FIELDS = {"dtc_code", "model_line", "module_name", "content"}


def load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def validate(items: object) -> list[tuple[str, str, str, str]]:
    if not isinstance(items, list):
        sys.exit("File must contain a JSON list.")
    rows = []
    for i, it in enumerate(items):
        if not isinstance(it, dict) or set(it) != FIELDS or not all(isinstance(it[k], str) for k in FIELDS):
            sys.exit(f"Item {i}: must be an object with exactly the string fields {sorted(FIELDS)}.")
        dtc, model = it["dtc_code"].strip().upper(), it["model_line"].strip().upper()
        name, content = it["module_name"].strip(), it["content"].strip()
        if not DTC.match(dtc) or not MODEL.match(model):
            sys.exit(f"Item {i}: invalid dtc_code or model_line.")
        if not (1 <= len(name) <= 200) or not (1 <= len(content) <= MAX_CONTENT):
            sys.exit(f"Item {i}: module_name must be 1-200 chars and content 1-{MAX_CONTENT} chars.")
        rows.append((dtc, model, name, content))
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rows = validate(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    load_env()
    with psycopg.connect(
        host="127.0.0.1", port=int(os.environ["POSTGRES_PORT"]), user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"], dbname=os.environ["POSTGRES_DB"],
    ) as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO manual_sections (dtc_code, model_line, module_name, content) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (dtc_code, model_line, module_name) DO UPDATE SET content = EXCLUDED.content",
            rows,
        )
        cur.execute("SELECT count(*) FROM manual_sections")
        print(f"imported {len(rows)} sections; table now has {cur.fetchone()[0]}")


if __name__ == "__main__":
    main()
