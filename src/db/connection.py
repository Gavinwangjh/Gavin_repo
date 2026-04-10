"""
Database connection utilities.

- Provides async context manager `session_factory()` for SQLAlchemy sessions.
- Use `async with session_factory() as session:` to safely open and close DB sessions.
- Central place to manage connection to the database.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings
from src.core.exceptions import DatabaseError

async_engine = create_async_engine(settings.async_db_url, pool_pre_ping=True, future=True)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def session_factory() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal(expire_on_commit=False) as session:
        try:
            yield session
        except SQLAlchemyError as e:
            await session.rollback()
            raise DatabaseError(message="Session error") from e
