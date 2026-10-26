"""Checks whether manual_chunk_summaries embeddings actually discriminate between sections, not just
whether retrieval runs.

The risk this targets (see the SYSTEM prompt in build_manual_summaries.py): the same DTC code can be
filed under several unrelated submodules (airbag module, radio module, actual battery module, ...) with
near-identical generic procedural text. If a summary's embedding doesn't capture what's actually
DIFFERENT about its section, a plain-language query will rank the wrong submodule's section highest -
and app/retrieval.py's semantic_evidence() would hand the tree-builder the wrong manual content.

Method: for every manual_chunk_summaries row, pull one of the LLM-written "likely questions" (the
second part of summary_for_search, after the blank line) as a query. Embed it, then rank it against
every OTHER chunk sharing at least one of the same dtc_codes + model_line (the actual competing set
semantic_evidence() would search over). A healthy embedding should put the chunk the question was
written for at rank 0 (or close), not behind a same-code rival from a different submodule.

Only chunks that share a code with at least one other chunk are evaluated - a code with a single
section can't be mis-ranked against anything, so it's not informative here.

Usage: uv run python scripts/eval_manual_summaries.py [--limit N] [--questions-per-chunk N] [--verbose]
"""
import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pgvector import Vector
from psycopg.rows import dict_row

from app.config import get_settings
from app.db import create_pool
from app.llm import OpenAILLM
from app.usage import UsageRecorder

OUT_PATH = Path(__file__).resolve().parent / "manual_summary_eval_results.json"
SEED = 42


def _questions(summary_for_search: str) -> list[str]:
    """The summary is: paragraph(s), a blank line, then one question per line. Split on the first
    blank line and return the question lines (drop anything too short to be a real question)."""
    parts = summary_for_search.split("\n\n", 1)
    if len(parts) < 2:
        return []
    lines = [ln.strip() for ln in parts[1].splitlines()]
    return [ln for ln in lines if len(ln) > 8]


async def fetch_chunks(pool) -> list[dict]:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT c.id, c.dtc_codes, c.model_line, c.submodule, s.summary_for_search "
            "FROM manual_chunks c JOIN manual_chunk_summaries s ON s.manual_chunk_id = c.id "
            "WHERE s.embedding IS NOT NULL"
        )
        return await cur.fetchall()


def group_by_rival_set(chunks: list[dict]) -> dict[int, set[int]]:
    """For each chunk id, the set of OTHER chunk ids it shares a (code, model_line) pair with -
    i.e. the real competing set semantic_evidence() ranks against when that code is given."""
    by_code_model: dict[tuple[str, str], list[int]] = {}
    for c in chunks:
        for code in c["dtc_codes"] or []:
            by_code_model.setdefault((code, c["model_line"]), []).append(c["id"])

    rivals: dict[int, set[int]] = {c["id"]: set() for c in chunks}
    for ids in by_code_model.values():
        for cid in ids:
            rivals[cid].update(i for i in ids if i != cid)
    return rivals


async def run_one(pool, llm, question: str, chunk_id: int, rival_ids: set[int]) -> dict:
    embedding = await llm.embed(question)
    vec = Vector(embedding)
    candidate_ids = [chunk_id, *rival_ids]
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT c.id, c.submodule, 1 - (s.embedding <=> %s) AS similarity "
            "FROM manual_chunk_summaries s JOIN manual_chunks c ON c.id = s.manual_chunk_id "
            "WHERE c.id = ANY(%s) ORDER BY s.embedding <=> %s",
            (vec, candidate_ids, vec),
        )
        ranked = await cur.fetchall()

    rank = next(i for i, r in enumerate(ranked) if r["id"] == chunk_id)
    top = ranked[0]
    return {
        "question": question,
        "target_chunk_id": chunk_id,
        "n_rivals": len(rival_ids),
        "rank": rank,  # 0 = correct chunk scored highest among its rivals
        "correct": rank == 0,
        "top_scored_chunk_id": top["id"],
        "top_scored_submodule": top["submodule"],
        "target_similarity": next(r["similarity"] for r in ranked if r["id"] == chunk_id),
        "top_similarity": top["similarity"],
    }


async def main(limit: int | None, questions_per_chunk: int, verbose: bool) -> None:
    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()
    llm = OpenAILLM(settings, UsageRecorder(pool))

    chunks = await fetch_chunks(pool)
    rivals = group_by_rival_set(chunks)
    contested = [c for c in chunks if rivals[c["id"]]]
    print(f"{len(chunks)} summarized chunk(s) total, {len(contested)} share a (code, model_line) with another chunk")

    rng = random.Random(SEED)
    cases = []
    for c in contested:
        qs = _questions(c["summary_for_search"])
        if not qs:
            continue
        picked = rng.sample(qs, min(questions_per_chunk, len(qs)))
        for q in picked:
            cases.append((c, q))

    if limit:
        cases = cases[:limit]
    print(f"{len(cases)} question(s) to check\n")

    results = []
    for i, (chunk, question) in enumerate(cases, 1):
        r = await run_one(pool, llm, question, chunk["id"], rivals[chunk["id"]])
        r["dtc_codes"] = chunk["dtc_codes"]
        r["model_line"] = chunk["model_line"]
        r["target_submodule"] = chunk["submodule"]
        results.append(r)
        mark = "OK  " if r["correct"] else "MISS"
        print(f"[{i}/{len(cases)}] {mark} rank={r['rank']}/{r['n_rivals']} "
              f"target={chunk['submodule']!r} vs top={r['top_scored_submodule']!r} :: {question[:70]}")
        if verbose and not r["correct"]:
            print(f"         codes={chunk['dtc_codes']} target_sim={r['target_similarity']:.4f} top_sim={r['top_similarity']:.4f}")

    await pool.close()

    n = len(results)
    correct = sum(r["correct"] for r in results)
    print("\n--- summary ---")
    print(f"correct (target ranked #1 among same-code rivals): {correct}/{n} ({100 * correct / n:.0f}%)" if n else "no cases")

    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\n{len(misses)} miss(es) - grouped by confused submodule pair:")
        pair_counts: dict[str, int] = {}
        for r in misses:
            key = f"{r['target_submodule']!r} -> {r['top_scored_submodule']!r}"
            pair_counts[key] = pair_counts.get(key, 0) + 1
        for pair, n_pair in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n_pair}x  {pair}")

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull detail written to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap total questions checked")
    parser.add_argument("--questions-per-chunk", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.questions_per_chunk, args.verbose), loop_factory=asyncio.SelectorEventLoop)
