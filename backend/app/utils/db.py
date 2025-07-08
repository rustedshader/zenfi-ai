import os
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker


engine = create_async_engine(
    sqlalchemy.engine.url.URL.create(
        drivername="postgresql+asyncpg",
        username=os.environ.get("APP_DB_USER", "postgres"),
        password=os.environ.get("APP_DB_PASS", "mysecretpassword"),
        host=os.environ.get("APP_INSTANCE_HOST", "localhost"),
        port=int(os.environ.get("APP_DB_PORT", "5432")),
        database=os.environ.get("APP_DB_NAME", "postgres"),
    ),
    echo=False,
)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
