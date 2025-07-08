import datetime
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UUID,
    Float,
    Date,
)
import sqlalchemy
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class FinanceNews(Base):
    __tablename__ = "finance_news"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime(timezone=True), index=True)
    news_data = Column(JSON, nullable=False)


# --- Pydantic Models ---


class NewChatInput(BaseModel):
    message: str
    isDeepSearch: bool


class ChatInput(BaseModel):
    session_id: str
    message: str
    isDeepSearch: bool


class ChatResponse(BaseModel):
    message: str
    sources: List = []


class StockInput(BaseModel):
    symbol: str = Field(..., min_length=1)


class StockSearchInput(BaseModel):
    input_query: str = Field(..., min_length=1)


class LinkBankAccountInput(BaseModel):
    bank_name: str


class BankAccountOutput(BaseModel):
    id: int
    bank_name: str
    masked_acc_number: str
    current_balance: float
    currency: str
    account_type: str
    status: str

    class Config:
        from_attributes = True


class TransactionOutput(BaseModel):
    id: int
    txn_id: str
    type: str
    mode: str
    amount: float
    current_balance: float
    transaction_timestamp: datetime.datetime
    value_date: datetime.date
    narration: str
    reference: str

    class Config:
        from_attributes = True


class AddTransactionInput(BaseModel):
    type: str
    mode: str
    amount: float
    current_balance: float
    narration: str
    reference: str
    transaction_timestamp: datetime.datetime
    value_date: datetime.date


class UpdateTransactionInput(BaseModel):
    type: Optional[str] = None
    mode: Optional[str] = None
    amount: Optional[float] = None
    current_balance: Optional[float] = None
    narration: Optional[str] = None
    reference: Optional[str] = None
    transaction_timestamp: Optional[datetime.datetime] = None
    value_date: Optional[datetime.date] = None


# --- New Pydantic Models for Portfolio ---


class AssetBase(BaseModel):
    asset_type: str
    identifier: str
    quantity: float
    purchase_price: float
    purchase_date: datetime.date
    notes: Optional[str] = None


class AssetCreate(AssetBase):
    pass


class AssetOutput(AssetBase):
    id: uuid.UUID
    portfolio_id: uuid.UUID
    created_at: datetime.datetime
    market_value: Optional[float] = None
    total_cost: Optional[float] = None
    profit_loss: Optional[float] = None
    percentage_change: Optional[float] = None
    stock_info: Optional[dict] = None
    news: Optional[list] = None

    class Config:
        from_attributes = True


class PortfolioBase(BaseModel):
    name: str
    description: Optional[str] = None
    gcs_document_link: Optional[str] = None
    is_default: bool = False


class PortfolioCreateInput(BaseModel):
    name: str
    description: Optional[str] = None
    is_default: Optional[bool] = False


class PortfolioCreate(PortfolioBase):
    pass


class PortfolioOutput(PortfolioBase):
    id: uuid.UUID
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class KnowledgeBaseCreateInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    meta_data: Optional[Dict[str, Any]] = None
    is_default: bool = False


class SetDefaultKnowledgeBaseInput(BaseModel):
    knowledge_base_id: str


class KnowledgeBaseOutput(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    meta_data: Optional[Dict[str, Any]]
    table_id: str
    is_default: bool
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class FileUploadResponse(BaseModel):
    message: str
    file_name: str
    status: str


class DocumentChunk(BaseModel):
    doc_id: str
    content: str
    embedding: List[float]
    meta_data: Dict[str, Any]


class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    filter: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    knowledge_base_id: str


class NewsBase(BaseModel):
    heading: str
    description: str
    content: str
    sources: str
