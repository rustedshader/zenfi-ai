# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run database migrations and start the application
CMD ["sh", "-c", "until pg_isready -h $APP_DB_HOST -p $APP_DB_PORT -U $APP_DB_USER; do echo 'Waiting for postgres...'; sleep 2; done; echo 'PostgreSQL is ready'; uv run alembic upgrade head && uv run uvicorn app.api.index:app --host 0.0.0.0 --port 8000"]
