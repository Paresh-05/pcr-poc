"""DTC manual content, read from the local manual_* tables (no external service call)."""
import re
from dataclasses import dataclass, field

from psycopg_pool import AsyncConnectionPool

TOTAL_CHARS = 60_000  # budget for all modules of one DTC together
MAX_CHARS_PER_MODULE = 8_000
MIN_CHARS_PER_MODULE = 1_500

_MID_PROCEDURE_START = re.compile(
    r"DTC Repair Instructions\s+(?:(?:Yes|No)\b|(?:[2-9]|[1-9]\d+)\.)",
    re.IGNORECASE,
)


@dataclass
class Section:
    module: str
    text: str


@dataclass
class ManualBundle:
    # used | not_found | unavailable | disabled
    status: str
    sections: list[Section] = field(default_factory=list)
    module_names: list[str] = field(default_factory=list)
    truncated: bool = False


def fit_budget(sections: list[Section]) -> tuple[list[Section], bool]:
    """Trim each module so that all of them together stay within TOTAL_CHARS."""
    if not sections:
        return [], False
    per_module = max(MIN_CHARS_PER_MODULE, min(MAX_CHARS_PER_MODULE, TOTAL_CHARS // len(sections)))
    truncated = False
    out = []
    for s in sections:
        if len(s.text) > per_module:
            s = Section(s.module, s.text[:per_module])
            truncated = True
        out.append(s)
    return out, truncated


def is_complete_procedure(text: str, dtc: str) -> bool:
    """True only for an end-to-end diagnostic procedure that is not visibly cut off."""
    compact = " ".join((text or "").split())
    head = compact[:500]
    completed_at = compact.lower().rfind("diagnosis completed")
    return bool(
        dtc.upper() in head.upper()
        and not _MID_PROCEDURE_START.search(head)
        and re.search(r"Circuit/System Inspection", compact, re.IGNORECASE)
        and re.search(r"\bYes\.", compact, re.IGNORECASE)
        and re.search(r"\bNo\.", compact, re.IGNORECASE)
        and completed_at >= 0
        and len(compact) - completed_at <= 500
    )


async def complete_manual_dtcs(pool: AsyncConnectionPool, model: str) -> set[str]:
    """DTCs with at least one complete manual procedure for this model mapping."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT d.dtc_code, s.content FROM manual_dtcs d "
            "JOIN manual_model_map m ON m.manual_model = d.manual_model "
            "JOIN manual_subsystems s ON s.manual_dtc_id = d.id "
            "WHERE m.model_line = %s",
            (model,),
        )
        rows = await cur.fetchall()
    return {dtc for dtc, content in rows if is_complete_procedure(content, dtc)}


async def has_complete_manual(pool: AsyncConnectionPool, dtc: str, model: str) -> bool:
    return dtc in await complete_manual_dtcs(pool, model)


async def load_manual(pool: AsyncConnectionPool, dtc: str, model: str) -> ManualBundle:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT d.manual_model, s.subsystem, s.content FROM manual_dtcs d "
            "JOIN manual_model_map m ON m.manual_model = d.manual_model "
            "JOIN manual_subsystems s ON s.manual_dtc_id = d.id "
            "WHERE d.dtc_code = %s AND m.model_line = %s ORDER BY d.manual_model, s.position",
            (dtc, model),
        )
        rows = await cur.fetchall()
    rows = [row for row in rows if is_complete_procedure(row[2], dtc)]
    if not rows:
        return ManualBundle("not_found")
    several_manuals = len({r[0] for r in rows}) > 1  # e.g. two Hector manuals: keep their sections apart
    sections = [Section(f"{mm} / {name}" if several_manuals else name, content) for mm, name, content in rows]
    return ManualBundle("used", sections, [s.module for s in sections], False)
