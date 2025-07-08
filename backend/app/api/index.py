from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import auth_router
from app.api.routes.chat import chat_router

app = FastAPI(
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "Endpoints for user authentication and authorization.",
        },
        {
            "name": "Chat",
            "description": "Endpoints for handling chat-related operations.",
        },
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api", tags=["Authentication"])
app.include_router(chat_router, prefix="/api", tags=["Chat"])
