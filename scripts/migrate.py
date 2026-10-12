"""Apply schema.sql to the running database (idempotent: every statement is CREATE ... IF NOT EXISTS)."""
import os
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent

for line in (ROOT / ".env").read_text().splitlines():
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

with psycopg.connect(
    host="127.0.0.1", port=int(os.environ["POSTGRES_PORT"]), user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"], dbname=os.environ["POSTGRES_DB"], autocommit=True,
) as conn:
    conn.execute((ROOT / "schema.sql").read_text())
    tables = [r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY 1")]
    print("schema applied; tables:", ", ".join(tables))
