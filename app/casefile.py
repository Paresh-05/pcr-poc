"""Everything known about a DTC for one diagnosis: the knowledge pack (built once per code) and the manual status."""
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from app import cases as case_repo
from app.llm import LLM
from app.manual import ManualBundle, load_manual
from app.pack import get_or_build_pack, render_pack


@dataclass
class Evidence:
    scope: str
    pack: dict
    manual: ManualBundle
    case_file: str  # the pack as text, stored with the diagnosis so later answers read the same facts

    @property
    def stats(self) -> dict:
        s = self.pack.get("stats", {})
        return {"total_cases": s.get("raw_cases", 0), "distinct_cases": s.get("distinct_cases", 0), "top_parts": s.get("top_parts", [])}

    @property
    def has_evidence(self) -> bool:
        return bool(self.pack.get("causes"))


async def gather_evidence(pool: AsyncConnectionPool, llm: LLM, dtc: str, model: str) -> Evidence:
    scope = await case_repo.resolve_scope(pool, dtc, model)
    manual = await load_manual(pool, dtc, model)
    if scope != case_repo.SCOPE_EXACT and manual.status != "used":
        return Evidence(scope, {}, manual, "")  # nothing to build a pack from: no AI call
    pack = await get_or_build_pack(pool, llm, dtc, model)
    return Evidence(scope, pack, manual, render_pack(pack, dtc, model))
