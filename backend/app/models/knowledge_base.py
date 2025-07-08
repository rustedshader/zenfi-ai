from datetime import datetime, timezone
import uuid
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import JSON, Index, UniqueConstraint, Column
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User


class KnowledgeBase(SQLModel, table=True):
    __tablename__ = "knowledge_bases"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    name: str = Field(index=True)
    description: str | None = Field(default=None)
    meta_data: dict | None = Field(default=None, sa_column=Column(JSON))
    table_id: str = Field(unique=True)
    is_default: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user: "User" = Relationship(back_populates="knowledge_bases")
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_user_knowledge_base_name"),
        Index(
            "uq_one_default_knowledge_base_per_user",
            "user_id",
            unique=True,
            postgresql_where="is_default",
        ),
    )
