"""Create/resize the embedding column and fill complaint embeddings. Resumable and idempotent.

Usage:
  uv run --system-certs python scripts/embed.py [--migrate-only] [--reset]

Reads from .env: EMBEDDING_API_KEY (or OPENAI_API_KEY), EMBEDDING_BASE_URL, EMBEDDING_MODEL,
EMBEDDING_DIMENSIONS (optional). The vector size is detected from the model and the DB column is
resized to match; if embeddings from a different size already exist, pass --reset to discard them.
Identical complaint texts are embedded once.
"""
import os
import sys
import time
from pathlib import Path

import psycopg
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import resolve_endpoint  # noqa: E402
from app.embeddings import MAX_CHARS, embedding_kwargs, to_vector_literal  # noqa: E402
from app.redact import redact  # noqa: E402

BATCH = 64
HNSW_MAX_DIMS = 2000  # pgvector's limit for HNSW on the `vector` type; above it we use exact scans


def load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def column_dim(conn) -> int:
    (dim,) = conn.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'cases'::regclass AND attname = 'complaint_embedding'"
    ).fetchone()
    return dim


def embed_batch(client: OpenAI, texts: list[str], kwargs: dict) -> list[list[float]]:
    for attempt in range(5):
        try:
            return [d.embedding for d in client.embeddings.create(input=texts, **kwargs).data]
        except Exception as exc:  # rate limits on free tiers: back off and retry
            if attempt == 4:
                raise
            wait = 5 * 2**attempt
            print(f"  retrying in {wait}s after {type(exc).__name__}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> None:
    load_env()
    migrate_only = "--migrate-only" in sys.argv
    reset = "--reset" in sys.argv
    model = os.environ.get("EMBEDDING_MODEL")
    if not model and not migrate_only:
        sys.exit("EMBEDDING_MODEL is not set in .env")
    dims_env = os.environ.get("EMBEDDING_DIMENSIONS")
    kwargs = embedding_kwargs(model, int(dims_env) if dims_env else None)

    with psycopg.connect(
        host="127.0.0.1", port=int(os.environ["POSTGRES_PORT"]), user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"], dbname=os.environ["POSTGRES_DB"], autocommit=True,
    ) as conn:
        conn.execute((ROOT / "migrations" / "001_embeddings.sql").read_text())
        print(f"migration applied (column size {column_dim(conn)})")
        if migrate_only:
            return

        key, base_url = resolve_endpoint(
            os.environ.get("EMBEDDING_API_KEY"),
            os.environ.get("EMBEDDING_BASE_URL") or None,
            os.environ.get("OPENROUTER_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
        )
        if not key:
            sys.exit("No EMBEDDING_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY set; cannot create embeddings.")
        client = OpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=2)

        # Detect the model's vector size and make the column match.
        dim = len(embed_batch(client, ["dimension probe"], kwargs)[0])
        current = column_dim(conn)
        print(f"model {model} returns {dim} dims; column is {current}")
        if dim != current:
            (existing,) = conn.execute("SELECT count(*) FROM cases WHERE complaint_embedding IS NOT NULL").fetchone()
            if existing and not reset:
                sys.exit(f"{existing} embeddings of size {current} exist; rerun with --reset to replace them.")
            conn.execute("DROP INDEX IF EXISTS idx_cases_complaint_embedding")
            conn.execute("UPDATE cases SET complaint_embedding = NULL")
            conn.execute(f"ALTER TABLE cases ALTER COLUMN complaint_embedding TYPE vector({dim})")
            print(f"column resized to {dim}")
        elif reset:
            conn.execute("UPDATE cases SET complaint_embedding = NULL")

        rows = conn.execute(
            "SELECT id, customer_verbatim FROM cases WHERE complaint_embedding IS NULL"
        ).fetchall()
        by_text: dict[str, list[int]] = {}
        for cid, text in rows:
            by_text.setdefault(redact(text)[:MAX_CHARS] or "unknown", []).append(cid)
        texts = list(by_text)
        print(f"{len(rows)} cases to embed, {len(texts)} distinct texts")

        for i in range(0, len(texts), BATCH):
            chunk = texts[i : i + BATCH]
            for text, vec in zip(chunk, embed_batch(client, chunk, kwargs)):
                if len(vec) != dim:
                    sys.exit("model returned inconsistent vector sizes")
                conn.execute(
                    "UPDATE cases SET complaint_embedding = %s::vector WHERE id = ANY(%s)",
                    (to_vector_literal(vec), by_text[text]),
                )
            print(f"embedded {min(i + BATCH, len(texts))}/{len(texts)}")

        if dim <= HNSW_MAX_DIMS:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cases_complaint_embedding "
                "ON cases USING hnsw (complaint_embedding vector_cosine_ops)"
            )
        else:
            print(f"{dim} dims exceeds the HNSW limit; skipping the index (exact search is fine at this size)")

        (n,) = conn.execute("SELECT count(*) FROM cases WHERE complaint_embedding IS NOT NULL").fetchone()
        print(f"cases with embeddings: {n}")


if __name__ == "__main__":
    main()
