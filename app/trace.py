"""Writes one clean, human-readable trace file per diagnosis build: the whole iteration (query, the
matched cases, the matched manual sections, the exact prompt sent to the model, and the tree it returned)
laid out in one place for easy before/after comparison across runs. This is separate from the regular
structured app log (logs/dtcbot.log), which stays interleaved with every other request and is much harder
to visually scan for one iteration.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.redact import redact
from app.schemas import DiagnosisTree

logger = logging.getLogger("dtcbot.trace")

TRACE_PATH = Path(__file__).resolve().parent.parent / "diagnosis_trace.log"


def _section(title: str) -> str:
    bar = "=" * 78
    return f"\n{bar}\n{title}\n{bar}\n"


def write_iteration_trace(
    *,
    diagnosis_id: str,
    dtc_code: str,
    model_line: str,
    query_text: str | None,
    cache_hit: bool,
    embedding: list[float] | None = None,
    cases: list[dict[str, Any]] | None = None,
    manual_chunks: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    tree: DiagnosisTree | None = None,
) -> None:
    """Overwrites diagnosis_trace.log at the project root with this one iteration. Never raises - a
    tracing failure must not break the actual diagnosis."""
    try:
        out: list[str] = []

        out.append(_section("DIAGNOSIS ITERATION TRACE"))
        out.append(f"Timestamp:      {datetime.now(timezone.utc).isoformat()}")
        out.append(f"Diagnosis id:   {diagnosis_id}")
        out.append(f"Cache:          {'HIT - reusing the tree already built for this code/model/symptom' if cache_hit else 'MISS - built fresh below'}")

        out.append(_section("1. USER QUERY"))
        out.append(f"DTC code given:   {dtc_code or '(none - matched by symptom text)'}")
        out.append(f"Model line:       {model_line}")
        out.append(f"Free-text query:  {redact(query_text) if query_text else '(none)'}")

        out.append(_section("2. QUERY EMBEDDING"))
        if embedding:
            out.append(f"Dimensions:      {len(embedding)}")
            out.append(f"First 8 values:  {embedding[:8]}")
            out.append("(the full vector is logged in logs/dtcbot.log under 'Query embedding' for this request)")
        elif cache_hit:
            out.append("(not recomputed - this result came from cache, see the run that first built it)")
        else:
            out.append("(not computed - exact code lookup, no embedding search needed)")

        cases = cases or []
        out.append(_section(f"3. PAST CASES FOUND ({len(cases)})"))
        if cache_hit and not cases:
            out.append("(not re-fetched - this result came from cache)")
        elif not cases:
            out.append("(none)")
        else:
            for i, c in enumerate(cases, 1):
                sim = c.get("similarity")
                out.append(f"--- Case {i}: ticket {c.get('ticket_no', '?')}{f' (similarity={sim:.3f})' if sim is not None else ''} ---")
                out.append(f"Complaint:  {redact(c.get('customer_verbatim', ''))}")
                out.append(f"Resolution: {redact(c.get('conclusion_text', ''))}")
                parts = ", ".join(p.get("part_name") or p.get("part_code", "") for p in (c.get("failed_parts") or []))
                out.append(f"Parts replaced: {parts or '(none)'}")
                out.append("")

        manual_chunks = manual_chunks or []
        out.append(_section(f"4. MANUAL SECTIONS FOUND ({len(manual_chunks)})"))
        if cache_hit and not manual_chunks:
            out.append("(not re-fetched - this result came from cache)")
        elif not manual_chunks:
            out.append("(none)")
        else:
            for i, m in enumerate(manual_chunks, 1):
                codes = ", ".join(m.get("dtc_codes") or []) or "?"
                out.append(f"--- Manual {i}: {m.get('submodule', '?')} (codes: {codes}) ---")
                out.append(m.get("full_markdown", ""))
                out.append("")

        out.append(_section("5. PROMPT SENT TO LLM - system message"))
        out.append(system_prompt if system_prompt else "(not recomputed - this result came from cache)")
        out.append(_section("5. PROMPT SENT TO LLM - user message"))
        out.append(user_prompt if user_prompt else "(not recomputed - this result came from cache)")

        out.append(_section("6. TREE OUTPUT"))
        out.append(json.dumps(tree.model_dump(), indent=2) if tree is not None else "(no tree - no_evidence)")

        out.append(_section("END OF TRACE"))

        TRACE_PATH.write_text("\n".join(out), encoding="utf-8")
        logger.info("Iteration trace written: %s", TRACE_PATH)
    except Exception:
        logger.exception("Failed to write iteration trace (non-fatal, diagnosis is unaffected)")
