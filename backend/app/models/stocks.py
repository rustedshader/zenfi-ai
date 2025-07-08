from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, Relationship
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User


class Stock(SQLModel, table=True):
    __tablename__ = "stocks"
    id: int = Field(default=None, primary_key=True, index=True)
    user_id: int = Field(foreign_key="users.id")
    symbol: str = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user: "User" = Relationship(back_populates="stocks")
