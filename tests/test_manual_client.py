"""Manual client behaviour against a mocked manual service (no network)."""
import asyncio

import httpx
from pydantic import SecretStr

from app import manual_client
from app.config import Settings
from app.manual_client import fetch_manual, html_to_text

HTML = '<html><body><script>alert(1)</script><div id="p1"><h2>P100016 test</h2><p>Check   connector X104</p></div></body></html>'


def _settings(**over) -> Settings:
    base = dict(
        api_key="k", postgres_user="u", postgres_password="p", postgres_db="d",
        manual_api_base_url="http://manual.test", manual_api_user="u", manual_api_password=SecretStr("p"),
    )
    return Settings.model_construct(**{**base, **over})


def _run(settings, handler, module=None, model="HECTOR"):
    manual_client._cache.clear()
    transport = httpx.MockTransport(handler)
    return asyncio.run(fetch_manual(settings, "P100016", model, module, transport=transport))


def _service(modules, calls=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(dict(request.url.params))
        if request.url.path.endswith("/modules"):
            return httpx.Response(200, json={"data": {"modules": [{"module_id": m["id"], "module_name": m["name"]} for m in modules]}})
        return httpx.Response(200, json={"data": {"pages": [HTML]}})

    return handler


def test_html_is_converted_to_plain_text_without_scripts():
    text = html_to_text(HTML)
    assert "alert" not in text and "<" not in text
    assert "Check connector X104" in text


def test_single_module_is_used_automatically():
    r = _run(_settings(), _service([{"id": 1, "name": "ECM"}]))
    assert r.status == "used" and r.module == "ECM" and "X104" in r.text


def test_multiple_modules_require_a_choice_then_resolve():
    mods = [{"id": 1, "name": "ECM"}, {"id": 2, "name": "TCM"}]
    r = _run(_settings(), _service(mods))
    assert r.status == "needs_module" and [m["name"] for m in r.modules] == ["ECM", "TCM"]
    assert _run(_settings(), _service(mods), module="tcm").module == "TCM"
    assert _run(_settings(), _service(mods), module="nope").status == "not_found"


def test_model_name_falls_back_to_spaces_when_underscore_unknown():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["model_name"])
        if request.url.params["model_name"] == "MG_ZS_EV":
            return httpx.Response(404)
        return _service([{"id": 1, "name": "BMS"}])(request)

    r = _run(_settings(), handler, model="MG_ZS_EV")
    assert r.status == "used" and seen[0] == "MG_ZS_EV" and "MG ZS EV" in seen


def test_not_found_unavailable_and_disabled():
    assert _run(_settings(), lambda r: httpx.Response(404)).status == "not_found"
    assert _run(_settings(), lambda r: httpx.Response(200, json={"data": {"modules": []}})).status == "not_found"

    def boom(request):
        raise httpx.ConnectError("down")

    assert _run(_settings(), boom).status == "unavailable"
    assert _run(_settings(manual_api_base_url=None), boom).status == "disabled"


def test_oversized_response_is_rejected_and_results_are_cached():
    big = lambda r: httpx.Response(200, content=b"x" * (manual_client.MAX_RESPONSE_BYTES + 1))  # noqa: E731
    assert _run(_settings(), big).status == "unavailable"

    calls: list = []
    manual_client._cache.clear()
    transport = httpx.MockTransport(_service([{"id": 1, "name": "ECM"}], calls))
    for _ in range(2):
        asyncio.run(fetch_manual(_settings(), "P100016", "HECTOR", None, transport=transport))
    assert len(calls) == 2  # modules + page once; second request served from cache
