from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from app.config import Settings


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)


def create_pool(settings: Settings) -> AsyncConnectionPool:
    return AsyncConnectionPool(settings.db_conninfo, min_size=1, max_size=10, open=False, configure=_configure)
