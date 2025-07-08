from pydantic import BaseModel
from typing import List, Optional


class Parts(BaseModel):
    type: str
    text: Optional[str] = None
    mediaType: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None


class ClientMessage(BaseModel):
    id: str
    role: str
    parts: List[Parts]
