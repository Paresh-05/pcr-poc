"""Query-time embedding of the technician's complaint."""
from openai import AsyncOpenAI

from app.config import Settings
from app.redact import redact

MAX_CHARS = 2000


def to_vector_literal(vec: list[float]) -> str:
    """pgvector text form, e.g. '[0.1,0.2]'. Only floats are formatted, so it is safe as a bound parameter."""
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"


def embedding_kwargs(model: str, dimensions: int | None) -> dict:
    kwargs: dict = {"model": model}
    if dimensions:
        kwargs["dimensions"] = dimensions
    return kwargs


async def embed_query(settings: Settings, text: str, expected_dim: int) -> str:
    """Embed the complaint; `expected_dim` is the DB column size, a mismatch raises (caller falls back)."""
    api_key, base_url = settings.embedding_endpoint
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=20,
        max_retries=1,
    )
    resp = await client.embeddings.create(
        input=redact(text)[:MAX_CHARS],
        **embedding_kwargs(settings.embedding_model, settings.embedding_dimensions),
    )
    vec = resp.data[0].embedding
    if len(vec) != expected_dim:
        raise ValueError("embedding size does not match database column")
    return to_vector_literal(vec)
