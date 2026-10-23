FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY run.py ./

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8100

CMD ["python", "run.py"]
