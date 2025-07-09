# Development Setup Instructions

## Prerequisites

- **Backend:** [uv](https://github.com/astral-sh/uv)
- **Frontend:** [Bun](https://bun.sh/)
- **Production:** [Docker](https://www.docker.com/)

---

## Local Development

### Backend

1. Navigate to the backend directory:
    ```bash
    cd backend
    ```
2. Install dependencies:
    ```bash
    uv sync
    ```
3. Activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```
4. Start the FastAPI development server:
    ```bash
    fastapi dev app/api/index.py
    ```

### Frontend

1. Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2. Install dependencies:
    ```bash
    bun i
    ```
3. Start the development server:
    ```bash
    bun run dev
    ```

---

## Production Setup

1. Ensure [Docker](https://www.docker.com/) is installed.
2. Build and start all services:
    ```bash
    docker-compose up -d --build
    ```

---


# Enviornment Variables

*Backend*

```bash
GEMINI_API_KEY
GOOGLE_API_KEY
GOOGLE_CX
```

*Frontend*
```bash
BACKEND_URL
NEXT_PUBLIC_API_URL
```

## Additional Notes

- Make sure to configure environment variables as needed for both backend and frontend.