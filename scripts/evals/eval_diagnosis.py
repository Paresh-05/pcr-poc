"""Runs scripts/golden_set.json (built by build_golden_set.py) through the real start_diagnosis
pipeline and checks whether the known-correct cause/parts actually surface in the generated tree -
as a root hypothesis, and as some leaf's conclusion. This is a sanity/regression check (the golden
query is itself retrievable from the corpus, see build_golden_set.py), meant to be rerun before/after
any change to TREE_SYSTEM, retrieval thresholds, or chunk caps so "did this help" is a number, not a
guess.

Usage: uv run python scripts/eval_diagnosis.py [--limit N] [--verbose]
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import diagnosis
from app.config import get_settings
from app.db import create_pool
from app.llm import OpenAILLM
from app.schemas import DiagnosisTree
from app.usage import UsageRecorder

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"
MATCH_THRESHOLD = 0.3  # fraction of expected-text tokens that must reappear to call it a match

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "was", "were", "it", "its", "for",
    "with", "at", "by", "as", "this", "that", "be", "has", "have", "had", "not", "no", "but", "if",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _overlap(expected: set[str], candidate: set[str]) -> float:
    if not expected:
        return 0.0
    return len(expected & candidate) / len(expected)


def _expected_tokens(case: dict) -> set[str]:
    return _tokens(case["expected_cause"]) | _tokens(" ".join(case["expected_parts"]))


def _hypothesis_match(tree: DiagnosisTree, expected: set[str]) -> dict:
    best = {"rank": None, "cause": None, "score": 0.0}
    for i, h in enumerate(tree.hypotheses):
        score = _overlap(expected, _tokens(h.cause))
        if score >= MATCH_THRESHOLD and (best["rank"] is None or score > best["score"]):
            best = {"rank": i, "cause": h.cause, "score": round(score, 2)}
    return best


def _leaf_match(tree: DiagnosisTree, expected: set[str]) -> dict:
    best = {"node_id": None, "problem": None, "confidence": None, "score": 0.0}
    for node in tree.nodes:
        if node.conclusion is None:
            continue
        candidate = _tokens(node.conclusion.problem) | _tokens(" ".join(node.conclusion.parts))
        score = _overlap(expected, candidate)
        if score >= MATCH_THRESHOLD and score > best["score"]:
            best = {
                "node_id": node.id, "problem": node.conclusion.problem,
                "confidence": node.conclusion.confidence, "score": round(score, 2),
            }
    return best


async def run_one(pool, llm, settings, case: dict, verbose: bool) -> dict:
    expected = _expected_tokens(case)
    try:
        result = await diagnosis.start_diagnosis(
            pool, llm, settings, case["model_line"], query_text=case["query_text"],
        )
    except Exception as exc:  # noqa: BLE001
        return {**case, "verdict": "error", "error": f"{type(exc).__name__}: {exc}"}

    if not result.get("tree"):
        return {**case, "verdict": "no_evidence", "scope": result["evidence"]["scope"]}

    tree = DiagnosisTree.model_validate(result["tree"])
    hyp = _hypothesis_match(tree, expected)
    leaf = _leaf_match(tree, expected)

    if leaf["node_id"] is not None:
        verdict = "found_in_leaf"
    elif hyp["rank"] is not None:
        verdict = "found_in_hypotheses_only"
    else:
        verdict = "not_found"

    out = {
        **case, "verdict": verdict, "scope": result["evidence"]["scope"],
        "cases_matched": result["evidence"]["total_cases"], "hypothesis_match": hyp, "leaf_match": leaf,
    }
    if verbose:
        out["all_hypotheses"] = [h.cause for h in tree.hypotheses]
    return out


async def main(limit: int | None, verbose: bool) -> None:
    if not GOLDEN_PATH.exists():
        print(f"no golden set found at {GOLDEN_PATH} - run scripts/build_golden_set.py first")
        return
    golden: list[dict[str, Any]] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if limit:
        golden = golden[:limit]

    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()
    llm = OpenAILLM(settings, UsageRecorder(pool))

    results = []
    for i, case in enumerate(golden, 1):
        r = await run_one(pool, llm, settings, case, verbose)
        results.append(r)
        print(f"[{i}/{len(golden)}] {case['dtc_code']} {case['ticket_no']}: {r['verdict']}")

    await pool.close()

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    total = len(results)
    print("\n--- summary ---")
    for verdict, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{verdict}: {n}/{total} ({100 * n / total:.0f}%)")

    out_path = Path(__file__).resolve().parent / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull detail written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.verbose), loop_factory=asyncio.SelectorEventLoop)
