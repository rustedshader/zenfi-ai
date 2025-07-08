from datetime import datetime, timezone
import uuid
from sqlmodel import SQLModel, Field, JSON


class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    user_id: int = Field(foreign_key="users.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str | None = Field(default=None)


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"
    id: int = Field(default=None, primary_key=True, index=True)
    session_id: uuid.UUID = Field(foreign_key="chat_sessions.id")
    sender: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: dict | None = Field(default=None, sa_type=JSON)
