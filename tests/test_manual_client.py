"""External manual client (not wired in for now) against a mocked service, no network."""
import asyncio

import httpx
from pydantic import SecretStr

from app import manual_client
from app.config import Settings
from app.manual_client import fetch_manual, html_to_text, reflow

HTML = '<html><body><script>alert(1)</script><div id="p1"><h2>P100016 test</h2><p>Check   connector X104</p></div></body></html>'
MODULES = [{"module_id": 1, "module_name": "ECM"}, {"module_id": 2, "module_name": "TCM"}]


def _settings(**over) -> Settings:
    base = dict(
        api_key="k", postgres_user="u", postgres_password="p", postgres_db="d",
        manual_api_base_url="http://manual.test/api", manual_api_user="u", manual_api_password=SecretStr("p"),
    )
    return Settings.model_construct(**{**base, **over})


def _run(settings, handler, model="HECTOR"):
    manual_client._cache.clear()
    return asyncio.run(fetch_manual(settings, "P100016", model, transport=httpx.MockTransport(handler)))


def _service(modules=MODULES, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request.url.path)
        if request.url.path.endswith("/modules"):
            return httpx.Response(200, json={"data": {"modules": modules}})
        return httpx.Response(200, json={"data": {"pages": [HTML]}})

    return handler


def test_html_becomes_plain_text_without_scripts():
    text = html_to_text(HTML)
    assert "alert" not in text and "<" not in text
    assert "Check connector X104" in text


def test_reflow_joins_words_but_keeps_steps_apart():
    raw = "11.\nTurn\nthe\nignition\noff.\n12.\nReplace\nthe\nmodule.\n–\nNo.\nDone.\n–\nYes.\nStop."
    assert reflow(raw).split("\n") == ["11. Turn the ignition off.", "12. Replace the module.", "– No.", "Done.", "– Yes.", "Stop."]


def test_all_modules_are_loaded_and_prefix_is_kept():
    seen: list[str] = []
    b = _run(_settings(), _service(seen=seen))
    assert b.status == "used" and b.module_names == ["ECM", "TCM"] and len(b.sections) == 2
    assert all(p.startswith("/api/v1/dtc/manual/") for p in seen)


def test_model_name_falls_back_to_spaces():
    names: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        names.append(request.url.params["model_name"])
        if request.url.params["model_name"] == "MG_ZS_EV":
            return httpx.Response(404)
        return _service()(request)

    assert _run(_settings(), handler, model="MG_ZS_EV").status == "used"
    assert names[0] == "MG_ZS_EV" and "MG ZS EV" in names


def test_not_found_unavailable_disabled_and_oversize():
    assert _run(_settings(), lambda r: httpx.Response(404)).status == "not_found"

    def boom(request):
        raise httpx.ConnectError("down")

    assert _run(_settings(), boom).status == "unavailable"
    assert _run(_settings(manual_api_base_url=None), boom).status == "disabled"
    big = lambda r: httpx.Response(200, content=b"x" * (manual_client.MAX_RESPONSE_BYTES + 1))  # noqa: E731
    assert _run(_settings(), big).status == "unavailable"
