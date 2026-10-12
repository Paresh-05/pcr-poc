"""DTC manual content, read from the local `manual_sections` table (no external service call)."""
from dataclasses import dataclass, field

from psycopg_pool import AsyncConnectionPool

TOTAL_CHARS = 60_000  # budget for all modules of one DTC together
MAX_CHARS_PER_MODULE = 8_000
MIN_CHARS_PER_MODULE = 1_500


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


async def load_manual(pool: AsyncConnectionPool, dtc: str, model: str) -> ManualBundle:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT module_name, content FROM manual_sections WHERE dtc_code = %s AND model_line = %s ORDER BY module_name",
            (dtc, model),
        )
        rows = await cur.fetchall()
    if not rows:
        return ManualBundle("not_found")
    sections, truncated = fit_budget([Section(name, content) for name, content in rows])
    return ManualBundle("used", sections, [s.module for s in sections], truncated)
