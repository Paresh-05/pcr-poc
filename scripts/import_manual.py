"""Load manual content into the database (idempotent).

Usage: uv run --system-certs python scripts/import_manual.py <dtc_all.json> [data/manual_model_map.json]

Input file: a JSON list; each item is one DTC of one manual model:
  {"model": "Hector MCE", "dtc_code": "P100016",
   "subsystems": [{"subsystem": "18.3 Engine Control Module 1.5T", "content": "plain text"}, ...]}
Map file: {"<manual model>": ["<our model_line>", ...]}; every model in the input must appear in it.
Re-running replaces the subsystems of the DTCs in the file and leaves other DTCs untouched.
"""
import json
import os
import re
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
DTC = re.compile(r"^[A-Z][0-9A-Z]{3,7}$")
MODEL_LINE = re.compile(r"^[A-Z0-9_]{2,32}$")
MAX_CONTENT = 200_000


def load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def fail(msg: str) -> None:
    sys.exit(f"Invalid input: {msg}")


def validate(items: object, model_map: dict) -> list[tuple[str, str, list[tuple[str, str]]]]:
    if not isinstance(items, list):
        fail("the file must contain a JSON list")
    out, seen = [], set()
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not {"model", "dtc_code", "subsystems"} <= set(it):
            fail(f"item {i} needs model, dtc_code and subsystems")
        model, dtc = str(it["model"]).strip(), str(it["dtc_code"]).strip().upper()
        if model not in model_map:
            fail(f"item {i}: manual model {model!r} is not in the map file")
        if not DTC.match(dtc):
            fail(f"item {i}: bad dtc_code {dtc!r}")
        if (model, dtc) in seen:
            fail(f"item {i}: duplicate {model!r} / {dtc}")
        seen.add((model, dtc))
        subs, names = [], set()
        for j, sub in enumerate(it["subsystems"] if isinstance(it["subsystems"], list) else []):
            name, content = str(sub.get("subsystem", "")).strip(), str(sub.get("content", "")).strip()
            if not (1 <= len(name) <= 200) or not (1 <= len(content) <= MAX_CONTENT) or name in names:
                fail(f"item {i} subsystem {j}: name must be unique, 1-200 chars; content 1-{MAX_CONTENT} chars")
            names.add(name)
            subs.append((name, content))
        if not subs:
            fail(f"item {i}: no subsystems")
        out.append((model, dtc, subs))
    return out


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    map_path = Path(sys.argv[2]) if len(sys.argv) == 3 else ROOT / "data" / "manual_model_map.json"
    model_map = json.loads(map_path.read_text(encoding="utf-8"))
    for manual_model, lines in model_map.items():
        if not isinstance(lines, list) or not all(isinstance(x, str) and MODEL_LINE.match(x) for x in lines):
            fail(f"map entry {manual_model!r} must be a list of model_line names like HECTOR")
    entries = validate(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), model_map)

    load_env()
    with psycopg.connect(
        host="127.0.0.1", port=int(os.environ["POSTGRES_PORT"]), user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"], dbname=os.environ["POSTGRES_DB"],
    ) as conn, conn.cursor() as cur:
        for manual_model, lines in model_map.items():
            for line in lines:
                cur.execute(
                    "INSERT INTO manual_model_map (manual_model, model_line) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (manual_model, line),
                )
        for model, dtc, subs in entries:
            cur.execute(
                "INSERT INTO manual_dtcs (manual_model, dtc_code) VALUES (%s, %s) "
                "ON CONFLICT (manual_model, dtc_code) DO UPDATE SET manual_model = EXCLUDED.manual_model RETURNING id",
                (model, dtc),
            )
            (dtc_id,) = cur.fetchone()
            cur.execute("DELETE FROM manual_subsystems WHERE manual_dtc_id = %s", (dtc_id,))
            cur.executemany(
                "INSERT INTO manual_subsystems (manual_dtc_id, position, subsystem, content) VALUES (%s, %s, %s, %s)",
                [(dtc_id, pos, name, content) for pos, (name, content) in enumerate(subs, start=1)],
            )
        cur.execute("SELECT (SELECT count(*) FROM manual_dtcs), (SELECT count(*) FROM manual_subsystems)")
        dtcs, subsystems = cur.fetchone()
    print(f"imported {len(entries)} DTC entries; database now has {dtcs} manual DTCs and {subsystems} subsystems")


if __name__ == "__main__":
    main()
