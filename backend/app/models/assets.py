from datetime import datetime, date, timezone
import uuid
from sqlmodel import SQLModel, Field, Relationship
from .portfolio import Portfolio


class Asset(SQLModel, table=True):
    __tablename__ = "assets"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    portfolio_id: uuid.UUID = Field(foreign_key="portfolios.id")
    asset_type: str
    identifier: str
    quantity: float
    purchase_price: float
    purchase_date: date
    current_value: float | None = Field(default=None)
    notes: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    portfolio: Portfolio = Relationship(back_populates="assets")
