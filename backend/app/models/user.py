from datetime import datetime, timezone
from typing import List
import uuid
from sqlmodel import SQLModel, Field, Relationship
from .stocks import Stock
from .portfolio import Portfolio
from .knowledge_base import KnowledgeBase


class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int = Field(default=None, primary_key=True, index=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stocks: List[Stock] = Relationship(
        back_populates="user", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

    portfolios: List[Portfolio] = Relationship(
        back_populates="user", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    knowledge_bases: List[KnowledgeBase] = Relationship(
        back_populates="user", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
