"""Clean the raw past-repair notes and the manual, and count exact numbers. Plain code, no AI.

The AI must never be asked to count: every number in the knowledge pack comes from here.
"""
import re
from collections import Counter
from dataclasses import dataclass, field

from app.cases import NO_PART, Case

MAX_NOTE_CHARS = 2500  # longest raw note is ~6k chars; the tail is almost always boilerplate

_GREETING = re.compile(
    r"^\s*(?:dear|respected|hello|hi|good (?:morning|afternoon|evening))\b[^.!\n]{0,60}[,.!:]?\s*"
    r"|greetings from [^.!\n]{0,80}[.!]*\s*"
    r"|(?:thanks|thank you|regards|warm regards|best regards)\b[^\n]{0,60}$",
    re.IGNORECASE,
)
_PAGE_MARK = re.compile(r"-{2,}\s*Page\s*\d+\s*-{2,}", re.IGNORECASE)
_DASHES = str.maketrans({"─": "-", "—": "-", "–": "-", "：": ":", " ": " "})
_PART_PREFIX = re.compile(r"^\d{6,}-")
_TOKEN = re.compile(r"[^a-z0-9]+")


def clean_note(text: str) -> str:
    """Whitespace, greetings and runs of dots/underscores removed; capped length."""
    text = (text or "").translate(_DASHES)
    text = re.sub(r"[.…_]{3,}", " ", text)  # '....!' style fillers
    text = _GREETING.sub(" ", text)
    text = " ".join(text.split()).lstrip(" !.,:;-")
    return text[:MAX_NOTE_CHARS]


def clean_manual(text: str) -> str:
    """Drop page markers, normalise odd dashes/colons, and rejoin lines broken in the middle of a sentence."""
    text = _PAGE_MARK.sub(" ", (text or "").translate(_DASHES))
    text = re.sub(r"(?<=[a-z,;(])\n(?=[a-z(])", " ", text)  # 'Circuit/System\nInspection' style breaks
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def part_name(code: str) -> str:
    """'000000000023574986-GASKET-EXH MUFF' -> 'GASKET-EXH MUFF'."""
    return _PART_PREFIX.sub("", (code or "").strip()).strip()


def _key(text: str) -> str:
    return _TOKEN.sub(" ", text.lower()).strip()[:300]


@dataclass
class CleanCase:
    id: int  # the first (lowest) case id of a group of near-identical notes
    note: str
    parts: list[str]  # plain part names, no "NOT APPLICABLE"
    no_part: bool
    copy_ids: list[int] = field(default_factory=list)  # every id in the group, including this one


@dataclass
class Prepared:
    raw_cases: int
    cases: list[CleanCase]  # de-duplicated
    part_counts: list[tuple[str, int]]  # (part name, distinct cases), most frequent first
    no_part_cases: int

    @property
    def distinct_cases(self) -> int:
        return len(self.cases)

    def pct(self, n: int) -> int:
        return round(100 * n / self.distinct_cases) if self.distinct_cases else 0

    def cases_with_parts(self, names: set[str]) -> tuple[int, int]:
        """(distinct notes, raw cases) whose parts include any of the given plain part names."""
        hit = [c for c in self.cases if names & set(c.parts)]
        return len(hit), sum(len(c.copy_ids) for c in hit)


def prepare_cases(cases: list[Case]) -> Prepared:
    groups: dict[str, CleanCase] = {}
    for c in sorted(cases, key=lambda c: c.id):
        note = clean_note(c.diagnosis)
        parts = sorted({part_name(p) for p in c.failed_parts if p != NO_PART})
        key = _key(note) or f"id{c.id}"
        if key in groups:
            groups[key].copy_ids.append(c.id)
            groups[key].parts = sorted(set(groups[key].parts) | set(parts))
            groups[key].no_part = groups[key].no_part and not parts
        else:
            groups[key] = CleanCase(c.id, note, parts, not parts, [c.id])
    distinct = list(groups.values())
    counts = Counter(p for c in distinct for p in c.parts)
    return Prepared(
        raw_cases=len(cases),
        cases=distinct,
        part_counts=sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])),
        no_part_cases=sum(c.no_part for c in distinct),
    )
