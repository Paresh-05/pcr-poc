"""Client for the DTC manual service (GET /v1/dtc/manual/modules and /v1/dtc/manual/page).

The service (mounted under /api, so set MANUAL_API_BASE_URL=http://host:3001/api) returns raw manual HTML, which is untrusted: it is converted to plain text here and only
ever shown to the LLM as data. This function never raises; failures become a status.
"""
import logging
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser

import httpx

from app.config import Settings

logger = logging.getLogger("dtcbot.manual")

MAX_RESPONSE_BYTES = 3_000_000
MAX_MANUAL_CHARS = 12_000
CACHE_TTL_SECONDS = 6 * 3600
MODULES_PATH = "/v1/dtc/manual/modules"
PAGE_PATH = "/v1/dtc/manual/page"

_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}
_SKIP_TAGS = {"script", "style", "iframe", "object", "embed", "noscript"}


@dataclass
class ManualResult:
    # used | needs_module | not_found | unavailable | disabled
    status: str
    text: str = ""
    module: str | None = None
    modules: list[dict] = field(default_factory=list)  # [{"id": ..., "name": ...}] for a picker


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


_cache: dict[tuple, tuple[float, ManualResult]] = {}


def _model_name_variants(model: str) -> list[str]:
    """Our data uses MG_ZS_EV; the manual service may know it as 'MG ZS EV'. Try both."""
    variants = [model]
    spaced = model.replace("_", " ")
    if spaced != model:
        variants.append(spaced)
    return variants


async def _get_json(client: httpx.AsyncClient, path: str, params: dict) -> tuple[int, object]:
    """GET with a hard response-size cap. Returns (status_code, parsed_json_or_None)."""
    async with client.stream("GET", path, params=params) as resp:
        if resp.status_code != 200:
            return resp.status_code, None
        body = b""
        async for chunk in resp.aiter_bytes():
            body += chunk
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("manual response too large")
        return 200, httpx.Response(200, content=body).json()


def _payload(body: object) -> object:
    return body.get("data", body) if isinstance(body, dict) else body


def _parse_modules(body: object) -> list[dict]:
    data = _payload(body)
    raw = data.get("modules") if isinstance(data, dict) else data
    modules = []
    for m in raw if isinstance(raw, list) else []:
        if not isinstance(m, dict):
            continue
        mid = m.get("module_id", m.get("id"))
        name = m.get("module_name", m.get("name", ""))
        if mid is not None or name:
            modules.append({"id": mid, "name": str(name)[:200]})
    return modules


def _parse_pages(body: object) -> str:
    data = _payload(body)
    pages = data.get("pages") if isinstance(data, dict) else None
    chunks = []
    for p in pages if isinstance(pages, list) else []:
        html = p if isinstance(p, str) else (p.get("html") or p.get("content") or "") if isinstance(p, dict) else ""
        chunks.append(html_to_text(html))
    return "\n\n".join(c for c in chunks if c)[:MAX_MANUAL_CHARS]


def _choose(modules: list[dict], wanted: str | None) -> dict | None:
    if wanted:
        w = wanted.strip().lower()
        for m in modules:
            if str(m["id"]).lower() == w or m["name"].lower() == w:
                return m
        return None
    return modules[0] if len(modules) == 1 else None


async def fetch_manual(
    settings: Settings, dtc: str, model: str, module: str | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> ManualResult:
    if not settings.manual_api_base_url:
        return ManualResult("disabled")

    key = (dtc, model, (module or "").lower())
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]

    auth = (
        httpx.BasicAuth(settings.manual_api_user, settings.manual_api_password.get_secret_value())
        if settings.manual_api_user and settings.manual_api_password
        else None
    )
    try:
        async with httpx.AsyncClient(
            base_url=settings.manual_api_base_url, auth=auth, timeout=15, follow_redirects=False, transport=transport
        ) as client:
            result = await _lookup(client, dtc, model, module)
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("manual lookup failed: %s", type(exc).__name__)
        return ManualResult("unavailable")

    if result.status in ("used", "not_found"):
        _cache[key] = (time.monotonic(), result)
    return result


async def _lookup(client: httpx.AsyncClient, dtc: str, model: str, module: str | None) -> ManualResult:
    modules: list[dict] = []
    name_used = model
    for name in _model_name_variants(model):
        status, body = await _get_json(client, MODULES_PATH, {"code": dtc, "model_name": name})
        if status == 200:
            modules = _parse_modules(body)
            name_used = name
            break
    if not modules:
        return ManualResult("not_found")

    chosen = _choose(modules, module)
    if chosen is None:
        if module:  # asked for a module that does not exist
            return ManualResult("not_found", modules=modules)
        return ManualResult("needs_module", modules=modules)

    status, body = await _get_json(
        client, PAGE_PATH, {"code": dtc, "model_name": name_used, "module": str(chosen["id"] if chosen["id"] is not None else chosen["name"])}
    )
    text = _parse_pages(body) if status == 200 else ""
    if not text:
        return ManualResult("not_found", modules=modules)
    return ManualResult("used", text=text, module=chosen["name"], modules=modules)
