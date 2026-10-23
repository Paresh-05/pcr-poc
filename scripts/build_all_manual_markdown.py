"""Generate the markdown doc for every DTC section found via the TOC, then concatenate them all into one file.

Reads scripts/sample_data/toc_sections.json (from build_toc_sections.py), runs build_manual_markdown's
generate_markdown() for each section with bounded concurrency, writes one .md file per section plus a
single concatenated file with everything, and a log of any sections that failed.

Usage: uv run python scripts/build_all_manual_markdown.py
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.llm import OpenAILLM
from build_manual_markdown import PDF_PATH, generate_markdown, section_filename  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_all_manual_markdown")

SECTIONS_PATH = Path(__file__).resolve().parent / "sample_data" / "toc_sections.json"
OUT_DIR = Path(__file__).resolve().parent / "sample_data" / "manual_markdown"
CONCAT_PATH = Path(__file__).resolve().parent / "sample_data" / "manual_complete.md"
CONCURRENCY = 8
MAX_RETRIES = 2


async def process_one(sem: asyncio.Semaphore, llm: OpenAILLM, reader: PdfReader, section: dict, idx: int, total: int) -> tuple[dict, str | None]:
    pages = list(range(section["start_page"], section["end_page"] + 1))
    name = section_filename(section["codes"], section["submodule"], section["start_page"])
    out_path = OUT_DIR / name

    if out_path.exists():  # resume support: skip ones already done
        return section, out_path.read_text(encoding="utf-8")

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                markdown = await generate_markdown(llm, reader, section["codes"], pages, section["model_line"])
                out_path.write_text(markdown, encoding="utf-8")
                log.info(f"[{idx}/{total}] OK   {name} ({section['page_count']}p, attempt {attempt})")
                return section, markdown
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[{idx}/{total}] FAIL {name} attempt {attempt}/{MAX_RETRIES}: {type(exc).__name__}: {exc}")
        return section, None


async def main(limit: int | None) -> None:
    settings = get_settings()
    llm = OpenAILLM(settings)  # no UsageRecorder: not writing to the DB in this step
    reader = PdfReader(str(PDF_PATH))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sections = json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
    if limit:
        sections = sections[:limit]
    total = len(sections)
    log.info(f"{total} sections to process, concurrency={CONCURRENCY}")

    sem = asyncio.Semaphore(CONCURRENCY)
    started = time.monotonic()
    tasks = [
        process_one(sem, llm, reader, section, i + 1, total)
        for i, section in enumerate(sections)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.monotonic() - started

    failed = [s["codes"][0] for s, md in results if md is None]
    ok = [(s, md) for s, md in results if md is not None]
    log.info(f"done in {elapsed:.0f}s: {len(ok)} ok, {len(failed)} failed")
    if failed:
        log.warning(f"failed sections: {failed}")

    # Concatenate in original TOC order.
    parts = []
    for section, markdown in zip(sections, (md for _, md in results)):
        if markdown is None:
            continue
        parts.append(
            f"<!-- submodule: {section['submodule']} | pages {section['start_page']}-{section['end_page']} -->\n\n"
            + markdown
        )
    CONCAT_PATH.write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    log.info(f"concatenated markdown written to {CONCAT_PATH} ({CONCAT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N sections (for testing)")
    args = parser.parse_args()
    asyncio.run(main(args.limit), loop_factory=asyncio.SelectorEventLoop)
