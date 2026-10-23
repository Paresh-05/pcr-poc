"""Build the full list of DTC sections from the manual's embedded table of contents.

Each level-4 TOC entry ("DTC <code1>, <code2>, ... or <codeN>") becomes one section. Its page range runs
from its own start page up to (not including) the next level-4 entry's start page - so a section always
includes every continuation page, even ones that don't repeat the DTC code. Its submodule is whichever
level-3 heading (e.g. "ABS DTC List") most recently preceded it in the TOC.

Output: scripts/sample_data/toc_sections.json - a list of
  {"codes": [...], "model_line": "MG_COMET_EV", "submodule": "...", "start_page": N, "end_page": M}
"""
import json
import re
from pathlib import Path

import pymupdf

PDF_PATH = Path(__file__).resolve().parent.parent / "MG_COMET_DTC_Diagnosis_Manual.pdf"
OUT_PATH = Path(__file__).resolve().parent / "sample_data" / "toc_sections.json"

_CODE_RE = re.compile(r"\b([PBCU][0-9][0-9A-Z]{3,6})\b", re.IGNORECASE)


def main() -> None:
    doc = pymupdf.open(str(PDF_PATH))
    toc = doc.get_toc()
    n_pages = doc.page_count

    current_submodule = None
    dtc_entries = []  # (codes, submodule, start_page)
    for level, title, page in toc:
        if level == 3:
            current_submodule = title.strip()
        elif level == 4 and title.strip().upper().startswith("DTC "):
            codes = [m.upper() for m in _CODE_RE.findall(title)]
            if codes:
                dtc_entries.append((codes, current_submodule, page))

    sections = []
    for i, (codes, submodule, start_page) in enumerate(dtc_entries):
        end_page = (dtc_entries[i + 1][2] - 1) if i + 1 < len(dtc_entries) else n_pages
        sections.append({
            "codes": codes,
            "model_line": "MG_COMET_EV",
            "submodule": submodule,
            "start_page": start_page,
            "end_page": end_page,
            "page_count": end_page - start_page + 1,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(sections, indent=2), encoding="utf-8")

    print(f"total DTC sections: {len(sections)}")
    print(f"total distinct codes covered: {len(set(c for s in sections for c in s['codes']))}")
    print(f"written to {OUT_PATH}")
    print()
    print("page_count distribution:")
    from collections import Counter
    dist = Counter(s["page_count"] for s in sections)
    for pc, n in sorted(dist.items()):
        print(f"  {pc} page(s): {n} section(s)")
    print()
    print("sample sections:")
    for s in sections[:3] + sections[-3:]:
        print(f"  {s['codes']} | {s['submodule']} | pages {s['start_page']}-{s['end_page']} ({s['page_count']}p)")


if __name__ == "__main__":
    main()
