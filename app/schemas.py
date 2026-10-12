from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, StringConstraints

_normalize = BeforeValidator(lambda v: v.strip().upper() if isinstance(v, str) else v)

DtcCode = Annotated[str, _normalize, StringConstraints(pattern=r"^[A-Z][0-9A-Z]{5,7}$")]
ModelLine = Annotated[str, _normalize, StringConstraints(pattern=r"^[A-Z0-9_]{2,32}$")]
Complaint = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)]


ManualModule = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[^\x00-\x1f\x7f]{1,200}$")]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtc_code: DtcCode
    model_line: ModelLine
    complaint: Complaint
    # Only needed when the manual has several control modules for this DTC (see response `manual.modules`).
    module: ManualModule | None = None
