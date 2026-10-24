"""Finds relevant past cases + manual procedures and turns them into evidence for the tree builder.

Two retrieval modes:
- exact: the catalog UI gives an exact (dtc_code, model_line) pair -> plain metadata filter, no embedding search
  needed (there is no free-text query to match by meaning).
- semantic: a free-text query -> filter by model (+code if present in the text), then rank by embedding
  similarity, dropping anything below the threshold. Not wired to a route yet; ready for a future free-text box.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any

from pgvector import Vector
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import Settings
from app.llm import LLM
from app.redact import redact

logger = logging.getLogger("dtcbot.retrieval")


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


MIN_CODE_CASES = 2  # a code needs at least this many matched cases...
MIN_CODE_SHARE = 0.2  # ...or this share of them, to have its manual pulled in
MANUAL_PAGES_PER_CODE = 3


def _supported_codes(cases: list[dict[str, Any]]) -> list[str]:
    """Codes among the matched cases with enough support to be a real candidate, most-supported first. The
    single most frequent code always qualifies, so a lone match still gets its manual."""
    counts: dict[str, int] = {}
    for c in cases:
        if c.get("dtc_code"):
            counts[c["dtc_code"]] = counts.get(c["dtc_code"], 0) + 1
    ranked = sorted(counts, key=counts.get, reverse=True)
    total = len(cases) or 1
    kept = [code for i, code in enumerate(ranked)
            if i == 0 or counts[code] >= MIN_CODE_CASES or counts[code] / total >= MIN_CODE_SHARE]
    dropped = {code: counts[code] for code in ranked if code not in kept}
    if dropped:
        logger.info("Manual lookup skips weakly supported codes (case counts): %s", dropped)
    return kept


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
    logger.info("Exact evidence lookup: dtc_code=%s model_line=%s (no embedding search)", dtc_code, model_line)
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
    logger.info(
        "Exact evidence result: dtc_code=%s model_line=%s cases=%d manual_chunks=%d scope=%s",
        dtc_code, model_line, len(cases), len(manual_chunks), scope,
    )
    return Evidence(cases=cases, manual_chunks=manual_chunks, top_parts=await _fix_counts(cases), scope=scope)


async def semantic_evidence(
    pool: AsyncConnectionPool, llm: LLM, settings: Settings, query_text: str, model_line: str, dtc_code: str | None
) -> Evidence:
    """Free-text flow: filter by model (+code if known), rank the rest by embedding similarity."""
    logger.info(
        "Semantic evidence lookup: model_line=%s dtc_code=%s query_len=%d threshold=%.2f top_k=%d",
        model_line, dtc_code, len(query_text), settings.similarity_threshold, settings.retrieval_top_k,
    )
    embedding = await llm.embed(query_text)
    logger.info("Query embedding (dims=%d) for query %s: %s", len(embedding), json.dumps(redact(query_text)), json.dumps(embedding))
    query_vec = Vector(embedding)

    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        if dtc_code:
            # The exact code match already makes every one of these cases relevant, so keep them all (same
            # 200 ceiling as exact_evidence) - the symptom only orders them, most similar first. Applying the
            # similarity cutoff here used to drop a 106-case code to 15 or even 0 just because the symptom
            # was worded casually, leaving a manual-only tree.
            await cur.execute(
                "SELECT c.ticket_no, c.customer_verbatim, c.conclusion_text, c.failed_parts, c.model_variant, "
                "c.fuel, c.updated_date, 1 - (s.embedding <=> %s) AS similarity FROM pcr_case_summaries s "
                "JOIN pcr_cases c ON c.id = s.pcr_case_id "
                "WHERE c.dtc_code = %s AND c.model_line = %s AND c.resolution_status = 'resolved' "
                "ORDER BY s.embedding <=> %s LIMIT 200",
                (query_vec, dtc_code, model_line, query_vec),
            )
            cases = await cur.fetchall()
            logger.info("Case lookup by code, ranked by symptom: kept all %d (no similarity cutoff)", len(cases))
        else:
            await cur.execute(
                "SELECT c.ticket_no, c.dtc_code, c.customer_verbatim, c.conclusion_text, c.failed_parts, "
                "c.model_variant, c.fuel, c.updated_date, 1 - (s.embedding <=> %s) AS similarity "
                "FROM pcr_case_summaries s JOIN pcr_cases c ON c.id = s.pcr_case_id "
                "WHERE c.model_line = %s AND c.resolution_status = 'resolved' "
                "ORDER BY s.embedding <=> %s LIMIT %s",
                (query_vec, model_line, query_vec, settings.retrieval_top_k),
            )
            ranked = await cur.fetchall()
            cases = [c for c in ranked if c["similarity"] >= settings.similarity_threshold][: settings.retrieval_keep]
            logger.info("Case similarity search: ranked=%d kept_after_threshold=%d", len(ranked), len(cases))

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
            # No code given: manual-summary embeddings don't reliably match casual symptom wording (a wrong
            # system's "vehicle may not start" outscored the right battery-fault section), so the manual is
            # chosen through the codes on the matched cases instead, which match accurately. Two guards:
            # only codes with real support among those cases count - one stray case shouldn't pull in a
            # whole system - and each code contributes only its few pages closest to the symptom, since a
            # code can span many (B100016 has 13 sections, ~76k chars) and would otherwise swamp the rest.
            manual_chunks = []
            supported = _supported_codes(cases)
            seen_ids: set[int] = set()
            for code in supported:
                await cur.execute(
                    "SELECT c.id, c.dtc_codes, c.submodule, c.full_markdown FROM manual_chunk_summaries s "
                    "JOIN manual_chunks c ON c.id = s.manual_chunk_id "
                    "WHERE %s = ANY(c.dtc_codes) AND c.model_line = %s "
                    "ORDER BY s.embedding <=> %s LIMIT %s",
                    (code, model_line, query_vec, MANUAL_PAGES_PER_CODE),
                )
                for row in await cur.fetchall():
                    if row["id"] not in seen_ids:
                        seen_ids.add(row["id"])
                        manual_chunks.append(row)
            logger.info(
                "Manual evidence (no code given): supported codes=%s -> %d manual chunk(s)",
                supported, len(manual_chunks),
            )

    scope = "semantic" if cases else ("manual_only" if manual_chunks else "none")
    logger.info(
        "Semantic evidence result: model_line=%s dtc_code=%s cases=%d manual_chunks=%d scope=%s",
        model_line, dtc_code, len(cases), len(manual_chunks), scope,
    )
    return Evidence(cases=cases, manual_chunks=manual_chunks, top_parts=await _fix_counts(cases), scope=scope)
