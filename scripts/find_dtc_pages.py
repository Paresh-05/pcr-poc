"""Scan the huge service-manual PDF for the given DTC codes and report which pages mention them."""
import re
import sys
import time
from pathlib import Path

from pypdf import PdfReader

CODES = ["B104A00", "B100016", "B104600"]
PDF_PATH = Path(__file__).resolve().parent.parent / "MG_COMET_DTC_Diagnosis_Manual.pdf"


def main() -> None:
    patterns = {c: re.compile(re.escape(c), re.IGNORECASE) for c in CODES}
    hits: dict[str, list[int]] = {c: [] for c in CODES}

    reader = PdfReader(str(PDF_PATH))
    n = len(reader.pages)
    print(f"total pages: {n}", flush=True)
    started = time.monotonic()
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            print(f"page {i+1}: extract error: {exc}", flush=True)
            continue
        for code, pat in patterns.items():
            if pat.search(text):
                hits[code].append(i + 1)
        if (i + 1) % 500 == 0:
            elapsed = time.monotonic() - started
            print(f"...scanned {i+1}/{n} pages ({elapsed:.0f}s elapsed)", flush=True)

    print()
    print("=== results ===")
    for code in CODES:
        pages = hits[code]
        print(f"{code}: {len(pages)} page(s) -> {pages[:30]}{'...' if len(pages) > 30 else ''}")


if __name__ == "__main__":
    main()
