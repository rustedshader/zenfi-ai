from datetime import datetime, timezone
import uuid
from sqlmodel import SQLModel, Field, Relationship
from typing import List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User
    from .assets import Asset


class Portfolio(SQLModel, table=True):
    __tablename__ = "portfolios"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    user_id: int = Field(foreign_key="users.id")
    name: str = Field(index=True)
    description: str | None = Field(default=None)
    gcs_document_link: str | None = Field(default=None)
    is_default: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user: "User" = Relationship(back_populates="portfolios")
    assets: List["Asset"] = Relationship(
        back_populates="portfolio",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
