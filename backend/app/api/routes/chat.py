from fastapi import APIRouter
from typing import List
from pydantic import BaseModel

from fastapi import Query
from fastapi.responses import StreamingResponse

from app.services.chat.normal.chat import ChatService
from app.types.api import ClientMessage

chat_router = APIRouter(prefix="/chat")


class Request(BaseModel):
    id: str
    messages: List[ClientMessage]
    trigger: str


@chat_router.post(path="/stream")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    print(request)
    messages = request.messages
    chat_service = ChatService()

    response = StreamingResponse(chat_service.stream_response(messages, protocol))
    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
