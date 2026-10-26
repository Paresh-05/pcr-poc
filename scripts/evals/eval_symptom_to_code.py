"""Checks whether the embedding-based retrieval (app/retrieval.py's semantic_evidence, no-code path)
actually surfaces the CORRECT DTC code from a free-text symptom alone - the real "mechanic types a
plain-language complaint, no code" scenario, exercising pcr_case_summaries embeddings end to end.

Uses scripts/evals/golden_set.json (real complaint -> real resolved dtc_code pairs, built by
build_golden_set.py). For each case, calls the real retrieval.semantic_evidence() with no dtc_code,
exactly as app/diagnosis.py does for a code-less query, and checks:
- top_case_correct: the single highest-similarity matched case belongs to the expected dtc_code.
- top_supported_code_correct: among cases that pass the similarity threshold, the dtc_code with the
  most matches (the one retrieval.py would actually pull manual pages for) is the expected one.

Caveat: each golden case is itself part of the retrieval corpus, so a very close textual duplicate of
itself scoring #1 is expected and not entirely "held out" - this is a regression/sanity check on the
retrieval mechanism, same caveat as build_golden_set.py documents for eval_diagnosis.py.

Usage: uv run python scripts/evals/eval_symptom_to_code.py [--limit N] [--verbose]
"""
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import retrieval
from app.config import get_settings
from app.db import create_pool
from app.llm import OpenAILLM
from app.usage import UsageRecorder

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"
OUT_PATH = Path(__file__).resolve().parent / "symptom_to_code_eval_results.json"


async def run_one(pool, llm, settings, case: dict) -> dict:
    ev = await retrieval.semantic_evidence(
        pool, llm, settings, case["query_text"], case["model_line"], dtc_code=None
    )
    expected = case["dtc_code"]

    top_case_code = ev.cases[0]["dtc_code"] if ev.cases else None
    code_counts = Counter(c["dtc_code"] for c in ev.cases if c.get("dtc_code"))
    top_supported_code = code_counts.most_common(1)[0][0] if code_counts else None

    return {
        "ticket_no": case["ticket_no"],
        "expected_code": expected,
        "query_text": case["query_text"],
        "scope": ev.scope,
        "n_cases_matched": len(ev.cases),
        "top_case_code": top_case_code,
        "top_case_correct": top_case_code == expected,
        "top_supported_code": top_supported_code,
        "top_supported_code_correct": top_supported_code == expected,
        "code_distribution": dict(code_counts.most_common(5)),
    }


async def main(limit: int | None, verbose: bool) -> None:
    if not GOLDEN_PATH.exists():
        print(f"no golden set found at {GOLDEN_PATH} - run scripts/evals/build_golden_set.py first")
        return
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if limit:
        golden = golden[:limit]

    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()
    llm = OpenAILLM(settings, UsageRecorder(pool))

    results = []
    for i, case in enumerate(golden, 1):
        r = await run_one(pool, llm, settings, case)
        results.append(r)
        mark = "OK  " if r["top_supported_code_correct"] else "MISS"
        print(f"[{i}/{len(golden)}] {mark} expected={r['expected_code']} top_case={r['top_case_code']} "
              f"top_supported={r['top_supported_code']} (n={r['n_cases_matched']}) :: {r['query_text'][:60]}")
        if verbose and not r["top_supported_code_correct"]:
            print(f"         code distribution: {r['code_distribution']}")

    await pool.close()

    n = len(results)
    top_case_ok = sum(r["top_case_correct"] for r in results)
    top_supported_ok = sum(r["top_supported_code_correct"] for r in results)
    print("\n--- summary ---")
    print(f"top single matched case has the correct code: {top_case_ok}/{n} ({100*top_case_ok/n:.0f}%)" if n else "no cases")
    print(f"the code retrieval.py would actually use (most-supported among matches) is correct: {top_supported_ok}/{n} ({100*top_supported_ok/n:.0f}%)" if n else "")

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull detail written to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.verbose), loop_factory=asyncio.SelectorEventLoop)
