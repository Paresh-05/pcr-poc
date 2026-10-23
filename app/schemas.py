from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints, model_validator

_normalize = BeforeValidator(lambda v: v.strip().upper() if isinstance(v, str) else v)

DtcCode = Annotated[str, _normalize, StringConstraints(pattern=r"^[A-Z][0-9][0-9A-Z]{3,6}$")]
ModelLine = Annotated[str, _normalize, StringConstraints(pattern=r"^[A-Z0-9_]{2,32}$")]
QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class StartDiagnosis(BaseModel):
    """Either dtc_code, or query_text (which may itself contain a DTC code), or both."""

    model_config = ConfigDict(extra="forbid")

    model_line: ModelLine
    dtc_code: DtcCode | None = None
    query_text: QueryText | None = None

    @model_validator(mode="after")
    def _require_one(self) -> "StartDiagnosis":
        if not self.dtc_code and not self.query_text:
            raise ValueError("either dtc_code or query_text is required")
        return self


class AnswerIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Literal["yes", "no", "not_sure"]


ShortText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)]
PartName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]


class OutcomeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: Literal["worked", "partly", "not_worked"]
    actual_cause: ShortText = ""
    parts_replaced: list[PartName] = Field(default_factory=list, max_length=10)
    notes: LongText = ""


# ---------- the diagnosis tree the LLM builds (one call, grounded in retrieved evidence) ----------

class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause: str
    likelihood: int = Field(ge=0, le=100)
    reasoning: str  # must cite its source, e.g. "manual: <module>" or "past cases (about N)"


class Alternative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause: str
    when_to_consider: str


class Conclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: str
    confidence: Literal["high", "medium", "low"]
    confidence_note: str
    evidence: list[str]
    fix_steps: list[str]
    parts: list[str]
    verify_fix: list[str]
    alternatives: list[Alternative]
    safety_note: str
    how_we_got_here: list[str] = Field(default_factory=list)
    ruled_out: list[str] = Field(default_factory=list)
    uncertain: list[str] = Field(default_factory=list)


class TreeNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    candidates: list[Candidate]
    # a leaf has conclusion set and question/yes/no/not_sure unset; a branch node has the opposite
    question: str | None = None
    yes: str | None = None
    no: str | None = None
    not_sure: str | None = None
    conclusion: Conclusion | None = None


class DiagnosisTree(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str
    nodes: list[TreeNode] = Field(max_length=31)  # depth ~5, binary-ish branching -> generous cap
