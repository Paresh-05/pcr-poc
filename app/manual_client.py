"""Client for the DTC manual service (GET {base}/v1/dtc/manual/modules and /page).

Set MANUAL_API_BASE_URL including the route prefix, e.g. http://localhost:3001/api.
The service returns raw manual HTML, which is untrusted: it is converted to plain text here and only ever shown
to the LLM as data. fetch_manual never raises; failures become a status.
"""
import asyncio
import logging
import re
import time
from html.parser import HTMLParser

import httpx

from app.config import Settings
from app.manual import ManualBundle, Section

logger = logging.getLogger("dtcbot.manual")

MAX_RESPONSE_BYTES = 3_000_000
TOTAL_CHARS = 60_000  # budget for all modules of one DTC together
MAX_CHARS_PER_MODULE = 8_000
MIN_CHARS_PER_MODULE = 1_500
CONCURRENCY = 4
CACHE_TTL_SECONDS = 6 * 3600
MODULES_PATH = "/v1/dtc/manual/modules"
PAGE_PATH = "/v1/dtc/manual/page"

_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}
_SKIP_TAGS = {"script", "style", "iframe", "object", "embed", "noscript"}


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


_NUMBERING = re.compile(r"^(\d+(\.\d+)*\.?|[–•ˆ-])$")


def reflow(text: str) -> str:
    """PDF-derived HTML arrives roughly one word per line; join lines back into paragraphs."""
    paragraphs: list[str] = []
    current = ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if current and not _NUMBERING.match(current) and (current[-1] in ".?!:：" or _NUMBERING.match(line)):
            paragraphs.append(current)
            current = line
        else:
            current = f"{current} {line}" if current else line
    if current:
        paragraphs.append(current)
    return "\n".join(paragraphs)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return reflow("\n".join(line for line in lines if line))


_cache: dict[tuple, tuple[float, ManualBundle]] = {}


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
    return "\n\n".join(c for c in chunks if c)


async def fetch_manual(
    settings: Settings, dtc: str, model: str, transport: httpx.AsyncBaseTransport | None = None
) -> ManualBundle:
    """All manual modules for this DTC and model, each trimmed so the total stays within TOTAL_CHARS."""
    if not settings.manual_api_base_url:
        return ManualBundle("disabled")

    key = (dtc, model)
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
            base_url=settings.manual_api_base_url, auth=auth, timeout=20, follow_redirects=False, transport=transport
        ) as client:
            bundle = await _lookup(client, dtc, model)
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("manual lookup failed: %s", type(exc).__name__)
        return ManualBundle("unavailable")

    if bundle.status in ("used", "not_found"):
        _cache[key] = (time.monotonic(), bundle)
    return bundle


async def _lookup(client: httpx.AsyncClient, dtc: str, model: str) -> ManualBundle:
    modules: list[dict] = []
    name_used = model
    for name in _model_name_variants(model):
        status, body = await _get_json(client, MODULES_PATH, {"code": dtc, "model_name": name})
        if status == 200:
            modules = _parse_modules(body)
            name_used = name
            break
    if not modules:
        return ManualBundle("not_found")

    per_module = max(MIN_CHARS_PER_MODULE, min(MAX_CHARS_PER_MODULE, TOTAL_CHARS // len(modules)))
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(m: dict) -> Section | None:
        async with sem:
            ref = str(m["id"] if m["id"] is not None else m["name"])
            status, body = await _get_json(client, PAGE_PATH, {"code": dtc, "model_name": name_used, "module": ref})
        text = _parse_pages(body) if status == 200 else ""
        return Section(m["name"], text) if text else None

    results = await asyncio.gather(*(one(m) for m in modules))
    sections, truncated = [], False
    for s in results:
        if s is None:
            continue
        if len(s.text) > per_module:
            s.text, truncated = s.text[:per_module], True
        sections.append(s)
    names = [m["name"] for m in modules]
    return ManualBundle("used", sections, names, truncated) if sections else ManualBundle("not_found", [], names)
