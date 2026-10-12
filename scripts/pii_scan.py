"""Count likely PII patterns in case text (prints counts only, never the text)."""
import os
import re
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

PATTERNS = {
    "phone_10digit": re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    "vin_17": re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
    "reg_plate": re.compile(r"\b[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{4}\b"),
    "url": re.compile(r"https?://\S+"),
}

with psycopg.connect(
    host="127.0.0.1", port=int(os.environ["POSTGRES_PORT"]), user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"], dbname=os.environ["POSTGRES_DB"],
) as conn:
    counts = dict.fromkeys(PATTERNS, 0)
    n = 0
    for diag, verb in conn.execute("SELECT diagnosis, customer_verbatim FROM cases"):
        n += 1
        text = f"{diag}\n{verb}"
        for name, rx in PATTERNS.items():
            if rx.search(text):
                counts[name] += 1
    print(n, counts)
