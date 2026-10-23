"""Pulls a DTC code out of raw free text, if one is typed inline (e.g. "getting P0420, car shakes")."""
import logging
import re

logger = logging.getLogger("dtcbot.query_analyzer")

# Letter prefix (P=Powertrain, B=Body, C=Chassis, U=Network), then a digit (real codes always have one
# here - this is what stops case-insensitive matching from treating plain words like "Problems" or
# "Battery" as codes), then 3-6 more alphanumeric characters: covers P0420 and this fleet's B100016 style.
_DTC_RE = re.compile(r"\b([PBCU][0-9][0-9A-Z]{3,6})\b", re.IGNORECASE)


def extract_dtc_code(text: str) -> tuple[str | None, str]:
    """Returns (code_or_None, remaining_text_with_the_code_removed)."""
    if not text:
        return None, ""
    match = _DTC_RE.search(text)
    if not match:
        logger.info("No DTC code found inline in free text (len=%d)", len(text))
        return None, text.strip()
    code = match.group(1).upper()
    remaining = (text[: match.start()] + text[match.end() :]).strip()
    remaining = re.sub(r"\s{2,}", " ", remaining)
    logger.info("Extracted DTC code %s from free text (remaining_len=%d)", code, len(remaining))
    return code, remaining
