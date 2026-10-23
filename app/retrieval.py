"""Finds relevant past cases + manual procedures and turns them into evidence for the tree builder.

Two retrieval modes:
- exact: the catalog UI gives an exact (dtc_code, model_line) pair -> plain metadata filter, no embedding search
  needed (there is no free-text query to match by meaning).
- semantic: a free-text query -> filter by model (+code if present in the text), then rank by embedding
  similarity, dropping anything below the threshold. Not wired to a route yet; ready for a future free-text box.
"""
from dataclasses import dataclass
from typing import Any

from pgvector import Vector
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import Settings
from app.llm import LLM


@dataclass(frozen=True)
class PartCount:
    part: str
    count: int
    pct: int


@dataclass(frozen=True)
class Evidence:
    cases: list[dict[str, Any]]
    manual_chunks: list[dict[str, Any]]
    top_parts: list[PartCount]
    scope: str  # "exact" | "semantic" | "manual_only" | "none"


async def _fix_counts(cases: list[dict[str, Any]]) -> list[PartCount]:
    counts: dict[str, int] = {}
    for case in cases:
        for part in case.get("failed_parts") or []:
            name = part.get("part_name") or part.get("part_code") or ""
            if name:
                counts[name] = counts.get(name, 0) + 1
    total = len(cases) or 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [PartCount(part=name, count=n, pct=round(100 * n / total)) for name, n in ranked]


async def exact_evidence(pool: AsyncConnectionPool, dtc_code: str, model_line: str) -> Evidence:
    """Used by the catalog flow: the DTC code is already exact, so no semantic search is needed."""
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT ticket_no, customer_verbatim, conclusion_text, failed_parts, model_variant, fuel, updated_date "
            "FROM pcr_cases WHERE dtc_code = %s AND model_line = %s AND resolution_status = 'resolved' "
            "ORDER BY updated_date DESC NULLS LAST LIMIT 200",
            (dtc_code, model_line),
        )
        cases = await cur.fetchall()

        await cur.execute(
            "SELECT id, submodule, full_markdown FROM manual_chunks "
            "WHERE %s = ANY(dtc_codes) AND model_line = %s",
            (dtc_code, model_line),
        )
        manual_chunks = await cur.fetchall()

    scope = "exact" if cases else ("manual_only" if manual_chunks else "none")
    return Evidence(cases=cases, manual_chunks=manual_chunks, top_parts=await _fix_counts(cases), scope=scope)


async def semantic_evidence(
    pool: AsyncConnectionPool, llm: LLM, settings: Settings, query_text: str, model_line: str, dtc_code: str | None
) -> Evidence:
    """Free-text flow: filter by model (+code if known), rank the rest by embedding similarity."""
    query_vec = Vector(await llm.embed(query_text))

    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        if dtc_code:
            await cur.execute(
                "SELECT c.ticket_no, c.customer_verbatim, c.conclusion_text, c.failed_parts, c.model_variant, "
                "c.fuel, c.updated_date, 1 - (s.embedding <=> %s) AS similarity FROM pcr_case_summaries s "
                "JOIN pcr_cases c ON c.id = s.pcr_case_id "
                "WHERE c.dtc_code = %s AND c.model_line = %s AND c.resolution_status = 'resolved' "
                "ORDER BY s.embedding <=> %s LIMIT %s",
                (query_vec, dtc_code, model_line, query_vec, settings.retrieval_top_k),
            )
        else:
            await cur.execute(
                "SELECT c.ticket_no, c.customer_verbatim, c.conclusion_text, c.failed_parts, c.model_variant, "
                "c.fuel, c.updated_date, 1 - (s.embedding <=> %s) AS similarity FROM pcr_case_summaries s "
                "JOIN pcr_cases c ON c.id = s.pcr_case_id "
                "WHERE c.model_line = %s AND c.resolution_status = 'resolved' "
                "ORDER BY s.embedding <=> %s LIMIT %s",
                (query_vec, model_line, query_vec, settings.retrieval_top_k),
            )
        ranked = await cur.fetchall()
        cases = [c for c in ranked if c["similarity"] >= settings.similarity_threshold][: settings.retrieval_keep]

        if dtc_code:
            # Filter by code first (a code can span many submodules - e.g. B100016 in 13), THEN rank by
            # the query's meaning so only the relevant submodule(s) go to the tree-builder. No threshold
            # drop here: the exact code match is already strong relevance signal on its own.
            await cur.execute(
                "SELECT c.id, c.submodule, c.full_markdown FROM manual_chunk_summaries s "
                "JOIN manual_chunks c ON c.id = s.manual_chunk_id "
                "WHERE %s = ANY(c.dtc_codes) AND c.model_line = %s "
                "ORDER BY s.embedding <=> %s LIMIT %s",
                (dtc_code, model_line, query_vec, settings.retrieval_keep),
            )
            manual_chunks = await cur.fetchall()
        else:
            await cur.execute(
                "SELECT c.id, c.submodule, c.full_markdown, "
                "1 - (s.embedding <=> %s) AS similarity FROM manual_chunk_summaries s "
                "JOIN manual_chunks c ON c.id = s.manual_chunk_id "
                "WHERE c.model_line = %s ORDER BY s.embedding <=> %s LIMIT %s",
                (query_vec, model_line, query_vec, settings.retrieval_top_k),
            )
            manual_rows = await cur.fetchall()
            manual_chunks = [m for m in manual_rows if m["similarity"] >= settings.similarity_threshold][: settings.retrieval_keep]

    scope = "semantic" if cases else ("manual_only" if manual_chunks else "none")
    return Evidence(cases=cases, manual_chunks=manual_chunks, top_parts=await _fix_counts(cases), scope=scope)
