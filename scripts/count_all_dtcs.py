"""Scan the full manual PDF and find every distinct DTC code actually documented in it (not just
mentioned), by matching the literal "DTC <code>" header pattern used throughout the book, plus rough
page-count stats so we can estimate real markdown-generation cost for the whole manual.
"""
import re
import sys
import time
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

PDF_PATH = Path(__file__).resolve().parent.parent / "MG_COMET_DTC_Diagnosis_Manual.pdf"
# "DTC" literally followed by a code - much more specific than the code appearing anywhere.
_DTC_HEADER_RE = re.compile(r"\bDTC\s+([PBCU][0-9][0-9A-Z]{3,6})\b")


def main() -> None:
    reader = PdfReader(str(PDF_PATH))
    n = len(reader.pages)
    print(f"total pages: {n}", flush=True)

    code_pages: dict[str, list[int]] = {}
    total_chars = 0
    started = time.monotonic()
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            print(f"page {i+1}: extract error: {exc}", flush=True)
            continue
        total_chars += len(text)
        for code in set(m.group(1).upper() for m in _DTC_HEADER_RE.finditer(text)):
            code_pages.setdefault(code, []).append(i + 1)
        if (i + 1) % 500 == 0:
            elapsed = time.monotonic() - started
            print(f"...scanned {i+1}/{n} pages ({elapsed:.0f}s elapsed, {len(code_pages)} distinct codes so far)", flush=True)

    print()
    print("=== results ===")
    print(f"distinct DTC codes found: {len(code_pages)}")
    print(f"total characters extracted across whole manual: {total_chars}")
    page_counts = Counter({c: len(p) for c, p in code_pages.items()})
    print()
    print("pages-per-code distribution (how many codes have N pages mentioning them):")
    dist = Counter(page_counts.values())
    for n_pages, n_codes in sorted(dist.items()):
        print(f"  {n_pages} page(s): {n_codes} code(s)")
    print()
    print("sample of codes with their page counts (first 30):")
    for code, pages in list(code_pages.items())[:30]:
        print(f"  {code}: {len(pages)} page(s)")


if __name__ == "__main__":
    main()
