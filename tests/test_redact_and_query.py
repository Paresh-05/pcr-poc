from app.redact import redact
from app.retrieval import build_tsquery


def test_redacts_private_details():
    text = "call 9876543210 or +91 9123456780, VIN MZBLG1234ABCD5678, plate MH12AB1234, a@b.com http://x.io/y"
    out = redact(text)
    for token in ("9876543210", "9123456780", "MZBLG1234ABCD5678", "MH12AB1234", "a@b.com", "http://x.io"):
        assert token not in out


def test_keeps_part_numbers_and_dtcs():
    text = "DTC P200200 part 000000000011640409 K68566622AA connector X104"
    assert redact(text) == text


def test_tsquery_is_alnum_or_only():
    q = build_tsquery("Car NOT starting!! ' | & ( ) :* ; DROP")
    assert all(ch.isalnum() or ch in " |" for ch in q)
    assert set(q.replace("|", " ").split()) <= {"car", "not", "starting", "drop"}
    assert build_tsquery("!!!") == ""


def test_llm_json_fallback_parses_fenced_json():
    from app.llm import _extract_json

    reply = (
        'Sure!\n```json\n{"summary":"s","likely_causes":[{"cause":"c","confidence":"low","evidence_case_ids":[1]}],'
        '"fix_steps":["a"],"parts_to_check":[],"data_sufficiency":"limited","safety_note":"n"}\n```'
    )
    assert _extract_json(reply).likely_causes[0].cause == "c"


def test_openrouter_key_never_goes_to_openai_endpoint():
    from app.config import OPENROUTER_BASE_URL, resolve_endpoint

    assert resolve_endpoint(None, None, "or-key", "oa-key") == ("or-key", OPENROUTER_BASE_URL)
    assert resolve_endpoint(None, "https://custom/v1", "or-key", None) == ("or-key", "https://custom/v1")
    assert resolve_endpoint("own", None, "or-key", "oa-key") == ("own", None)
    assert resolve_endpoint(None, None, None, "oa-key") == ("oa-key", None)
    assert resolve_endpoint(None, None, None, None) == (None, None)


def test_chat_prefers_direct_openai_and_embeddings_prefer_openrouter():
    from app.config import OPENROUTER_BASE_URL, resolve_endpoint

    both = ("or-key", "oa-key")
    assert resolve_endpoint(None, None, *both, prefer="openai") == ("oa-key", None)  # chat -> OpenAI direct
    assert resolve_endpoint(None, None, *both, prefer="openrouter") == ("or-key", OPENROUTER_BASE_URL)  # embeddings
    assert resolve_endpoint(None, None, "or-key", None, prefer="openai") == ("or-key", OPENROUTER_BASE_URL)


def test_openrouter_style_key_never_defaults_to_openai_endpoint():
    from app.config import OPENROUTER_BASE_URL, resolve_endpoint

    assert resolve_endpoint("sk-or-v1-abc", None, None, None) == ("sk-or-v1-abc", OPENROUTER_BASE_URL)
    assert resolve_endpoint("sk-proj-abc", None, None, None) == ("sk-proj-abc", None)
