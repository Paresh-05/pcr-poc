"""Pushes the regenerated manual markdown files (scripts/sample_data/manual_markdown/) into the
DATABASE - UPDATEs manual_chunks.full_markdown for each existing row, matched by
(model_line, submodule, start_page), same key used by load_manual_markdown_only.py.

This does NOT touch manual_chunk_summaries or its embeddings - those were written from the OLD
full_markdown and are now stale (the AI-invented analysis sections are gone, headings changed).
Re-run scripts/build_manual_summaries.py / scripts/embed_manual_summaries.py afterward to refresh
them, or pass --reset-summaries here to delete the now-stale summary rows so the next run of
build_manual_summaries.py regenerates them.

Usage: uv run python scripts/update_manual_markdown_db.py [--limit N] [--dry-run] [--reset-summaries]
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import create_pool
from build_manual_markdown import section_filename  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("update_manual_markdown_db")

SECTIONS_PATH = Path(__file__).resolve().parent / "sample_data" / "toc_sections.json"
MARKDOWN_DIR = Path(__file__).resolve().parent / "sample_data" / "manual_markdown"


async def main(limit: int | None, dry_run: bool, reset_summaries: bool) -> None:
    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()

    sections = json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
    if limit:
        sections = sections[:limit]
    total = len(sections)
    log.info(f"{total} sections to check (dry_run={dry_run}, reset_summaries={reset_summaries})")

    updated = unchanged = missing_file = missing_row = 0
    for i, section in enumerate(sections, 1):
        name = section_filename(section["codes"], section["submodule"], section["start_page"])
        md_path = MARKDOWN_DIR / name
        if not md_path.exists():
            log.warning(f"[{i}/{total}] SKIP {name}: regenerated markdown file not found")
            missing_file += 1
            continue

        new_markdown = md_path.read_text(encoding="utf-8")

        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, full_markdown FROM manual_chunks WHERE model_line = %s AND submodule = %s AND start_page = %s LIMIT 1",
                (section["model_line"], section["submodule"], section["start_page"]),
            )
            row = await cur.fetchone()
            if row is None:
                log.warning(f"[{i}/{total}] SKIP {name}: no manual_chunks row found for this section")
                missing_row += 1
                continue
            chunk_id, old_markdown = row

            if old_markdown == new_markdown:
                unchanged += 1
                continue

            if dry_run:
                updated += 1
                continue

            await conn.execute(
                "UPDATE manual_chunks SET full_markdown = %s WHERE id = %s", (new_markdown, chunk_id)
            )
            if reset_summaries:
                await conn.execute(
                    "DELETE FROM manual_chunk_summaries WHERE manual_chunk_id = %s", (chunk_id,)
                )
            updated += 1
            if updated % 50 == 0:
                log.info(f"[{i}/{total}] ...updated {updated} so far")

    log.info(
        f"done: {updated} updated{' (dry-run, no writes)' if dry_run else ''}, {unchanged} already matched, "
        f"{missing_file} missing file, {missing_row} missing db row"
    )
    await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    parser.add_argument("--reset-summaries", action="store_true",
                         help="also delete manual_chunk_summaries rows for updated chunks (so they get regenerated from the new markdown)")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.dry_run, args.reset_summaries), loop_factory=asyncio.SelectorEventLoop)
