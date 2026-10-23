"""Launcher. Windows needs a selector event loop for psycopg async; Linux (Docker) doesn't."""
import asyncio
import os
import sys

import uvicorn


def main() -> None:
    host = os.environ.get("APP_HOST", "127.0.0.1")
    config = uvicorn.Config("app.main:app", host=host, port=8100)
    kwargs = {"loop_factory": asyncio.SelectorEventLoop} if sys.platform == "win32" else {}
    asyncio.run(uvicorn.Server(config).serve(), **kwargs)


if __name__ == "__main__":
    main()
