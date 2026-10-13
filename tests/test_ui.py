"""The demo page: strict CSP, no key in the browser, localhost-only or passcode access. No database needed."""
import re
import time
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import Settings, get_settings
from app.main import app
from app.ui_access import COOKIE, make_token, token_is_valid

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
LOCAL = ("127.0.0.1", 50000)
REMOTE = ("203.0.113.9", 50000)


def _settings(passcode: str | None = None) -> Settings:
    return Settings.model_construct(
        app_env="dev", api_key=SecretStr("test-api-key-0123456789"), rate_limit_per_minute=1000,
        demo_passcode=SecretStr(passcode) if passcode else None,
    )


def _client(client=LOCAL, passcode: str | None = None) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: _settings(passcode)
    return TestClient(app, client=client, raise_server_exceptions=False)  # lifespan not started: no database here


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_page_is_served_with_a_strict_csp_and_needs_no_key():
    res = _client().get("/ui/")
    assert res.status_code == 200 and "DTC" in res.text
    csp = res.headers["content-security-policy"]
    for directive in ("default-src 'self'", "script-src 'self'", "object-src 'none'", "frame-ancestors 'none'", "base-uri 'none'"):
        assert directive in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "*" not in csp
    assert res.headers["x-content-type-options"] == "nosniff" and res.headers["cache-control"] == "no-store"
    assert "api-key" not in res.text.lower() and "x-api-key" not in (STATIC / "app.js").read_text(encoding="utf-8").lower()


def test_without_a_passcode_only_this_computer_is_served():
    assert _client(LOCAL).get("/ui-api/session").json() == {"passcode_required": False, "authenticated": True}
    assert _client(REMOTE).get("/ui-api/session").json() == {"passcode_required": False, "authenticated": False}
    assert _client(REMOTE).get("/ui-api/catalog").status_code == 403
    assert _client(REMOTE).post("/ui-api/diagnoses", json={"dtc_code": "P187973", "model_line": "HECTOR"}).status_code == 403


def test_trusting_the_network_serves_the_page_without_a_code_but_a_passcode_still_wins():
    def trusted(passcode=None):
        app.dependency_overrides[get_settings] = lambda: _settings(passcode).model_copy(update={"demo_trust_network": True})
        return TestClient(app, client=REMOTE, raise_server_exceptions=False)

    assert trusted().get("/ui-api/session").json() == {"passcode_required": False, "authenticated": True}
    assert trusted().get("/ui-api/catalog").status_code not in (401, 403)  # allowed (fails later: no database here)
    assert trusted("secret").get("/ui-api/session").json() == {"passcode_required": True, "authenticated": False}
    assert trusted().get("/catalog").status_code == 401  # the programmatic API is unaffected


def test_the_programmatic_api_still_needs_the_api_key():
    c = _client(LOCAL)  # being on localhost does not bypass the key on the normal API
    for method, path in (("get", "/catalog"), ("get", "/outcomes/summary"), ("post", "/diagnoses")):
        assert getattr(c, method)(path).status_code == 401, path


def test_passcode_login_sets_a_signed_httponly_cookie():
    c = _client(REMOTE, passcode="correct horse")
    assert c.get("/ui-api/session").json() == {"passcode_required": True, "authenticated": False}
    assert c.get("/ui-api/catalog").status_code == 401
    assert c.post("/ui-api/session", json={"passcode": "wrong"}).status_code == 401

    ok = c.post("/ui-api/session", json={"passcode": "correct horse"})
    assert ok.status_code == 200 and ok.json()["authenticated"] is True
    cookie = ok.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/ui-api" in cookie
    assert c.get("/ui-api/session").json()["authenticated"] is True
    assert c.get("/ui-api/catalog").status_code not in (401, 403)  # authorised (then fails later: no database in this test)


def test_cross_origin_posts_are_refused():
    c = _client(LOCAL)
    assert c.post("/ui-api/diagnoses", json={"dtc_code": "P187973", "model_line": "HECTOR"}, headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.post("/ui-api/session", json={"passcode": "x"}, headers={"Origin": "http://evil.example"}).status_code == 403


def test_tokens_expire_and_cannot_be_forged():
    s = _settings("pw")
    token = make_token(s)
    assert token_is_valid(s, token)
    exp, sig = token.split(".")
    assert not token_is_valid(s, f"{int(exp) + 1}.{sig}") and not token_is_valid(s, f"{exp}.{'0' * len(sig)}")
    assert not token_is_valid(s, f"{int(time.time()) - 5}.{sig}") and not token_is_valid(s, None) and not token_is_valid(s, "garbage")
    assert not token_is_valid(_settings("pw").model_copy(update={"api_key": SecretStr("another-secret-value")}), token)
    assert COOKIE == "dtc_ui"


def test_page_has_no_inline_code_third_party_resources_or_unsafe_dom_calls():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html) and not re.search(r"\son\w+\s*=", html) and "style=" not in html
    assert not re.findall(r'(?:src|href)="(?:https?:)?//', html)
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "localStorage", "sessionStorage", "document.cookie"):
        assert banned not in js, banned
