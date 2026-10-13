from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints

_normalize = BeforeValidator(lambda v: v.strip().upper() if isinstance(v, str) else v)

DtcCode = Annotated[str, _normalize, StringConstraints(pattern=r"^[A-Z][0-9A-Z]{5,7}$")]
ModelLine = Annotated[str, _normalize, StringConstraints(pattern=r"^[A-Z0-9_]{2,32}$")]
Note = Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)]


class StartDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtc_code: DtcCode
    model_line: ModelLine


class AnswerIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Literal["yes", "no", "not_sure"]
    note: Note = ""


ShortText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)]
PartName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]


class OutcomeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: Literal["worked", "partly", "not_worked"]
    actual_cause: ShortText = ""
    parts_replaced: list[PartName] = Field(default_factory=list, max_length=10)
    notes: LongText = ""
