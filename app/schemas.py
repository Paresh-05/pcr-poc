from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, StringConstraints

_normalize = BeforeValidator(lambda v: v.strip().upper() if isinstance(v, str) else v)

DtcCode = Annotated[str, _normalize, StringConstraints(pattern=r"^[A-Z][0-9A-Z]{5,7}$")]
ModelLine = Annotated[str, _normalize, StringConstraints(pattern=r"^[A-Z0-9_]{2,32}$")]
MessageText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class StartConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtc_code: DtcCode
    model_line: ModelLine


class UserMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: MessageText
