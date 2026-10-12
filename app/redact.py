"""Strip private details from text before it is returned or sent to the LLM."""
import re

_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"https?://\S+"), "[URL]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"), "[EMAIL]"),
    (
        re.compile(
            r"(?<![\w])(?=[A-HJ-NPR-Z0-9]*\d)(?=[A-HJ-NPR-Z0-9]*[A-HJ-NPR-Z])[A-HJ-NPR-Z0-9]{17}(?![\w])"
        ),
        "[VIN]",
    ),
    (re.compile(r"\b[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{4}\b"), "[PLATE]"),
    (re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{9}(?!\d)"), "[PHONE]"),
]


def redact(text: str) -> str:
    for rx, repl in _RULES:
        text = rx.sub(repl, text)
    return text
