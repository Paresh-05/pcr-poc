"""The "case file": everything known about a DTC (past cases, stats, complaints, manual), built once per diagnosis."""
import asyncio
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from app import cases as case_repo
from app.digest import get_digest
from app.llm import LLM
from app.manual import ManualBundle, load_manual


@dataclass
class Evidence:
    scope: str
    stats: dict
    manual: ManualBundle
    case_file: str

    @property
    def has_evidence(self) -> bool:
        return self.stats["total_cases"] > 0 or self.manual.status == "used"


def build_case_file(dtc: str, model: str, stats: dict, complaints: list[tuple[str, int]], digest: str, manual: ManualBundle) -> str:
    scope_text = {
        case_repo.SCOPE_EXACT: f"past cases for this DTC on {model}",
        case_repo.SCOPE_ALL_MODELS: f"NO cases on {model}; showing this DTC on other models",
        case_repo.SCOPE_NONE: "no past cases for this DTC",
    }[stats["scope"]]
    lines = [f"DTC: {dtc}", f"Model: {model}", f"Case scope: {scope_text}", f"Total past cases: {stats['total_cases']}"]
    if stats["total_cases"]:
        lines.append(f"Cases where no part was replaced: {stats['no_part_replaced_pct']}%")
        lines.append("Parts most often replaced:")
        lines += [f"- {p['part']}: {p['cases']} cases ({p['pct']}%)" for p in stats["top_parts"]] or ["- none recorded"]
        lines.append("Most common customer complaints:")
        lines += [f'- "{text}": {n} cases' for text, n in complaints]
    lines += ["", "Digest of all past cases:", digest, ""]
    if manual.status == "used":
        lines.append("Official manual sections" + (" (trimmed to fit)" if manual.truncated else "") + ":")
        lines += [f"=== MODULE: {s.module} ===\n{s.text}" for s in manual.sections]
    else:
        lines.append("Official manual: not available for this DTC and model.")
    return "\n".join(lines)


async def gather_evidence(pool: AsyncConnectionPool, llm: LLM, dtc: str, model: str) -> Evidence:
    scope = await case_repo.resolve_scope(pool, dtc, model)
    cases = await case_repo.load_cases(pool, dtc, model, scope)
    stats, complaints, manual = await asyncio.gather(
        case_repo.compute_stats(pool, dtc, model, scope),
        case_repo.complaint_groups(pool, dtc, model, scope),
        load_manual(pool, dtc, model),
    )
    if stats["total_cases"] == 0 and manual.status != "used":
        return Evidence(scope, stats, manual, "")
    digest = await get_digest(pool, llm, dtc, model, scope, cases)
    return Evidence(scope, stats, manual, build_case_file(dtc, model, stats, complaints, digest, manual))
