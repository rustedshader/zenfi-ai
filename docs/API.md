# Zenfi AI API Documentation

This document provides comprehensive information about the Zenfi AI API endpoints and their usage.

## Base URL

```
http://localhost:8000
```

## Authentication

Zenfi AI uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## API Endpoints

### Authentication

#### POST /api/auth/register
Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "message": "User registered successfully",
  "user": {
    "id": "integer",
    "username": "string",
    "email": "string"
  }
}
```

#### POST /api/auth/login
Login with existing credentials.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "user": {
    "id": "integer",
    "username": "string",
    "email": "string"
  }
}
```

#### POST /api/auth/logout
Logout the current user.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "message": "Logged out successfully"
}
```

### Chat

#### POST /api/chat/stream
Stream chat responses with the AI assistant.

**Headers:**
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "id": "string",
  "messages": [
    {
      "role": "user",
      "content": "string",
      "parts": [
        {
          "type": "text",
          "text": "string"
        }
      ]
    }
  ],
  "trigger": "string"
}
```

**Response:**
Streaming response with Server-Sent Events (SSE) format.

## Available Financial Tools

The AI assistant has access to various financial tools through the chat interface:

### Stock Information Tools

- **get_stock_currency(symbol)** - Get stock currency
- **get_stock_day_high(symbol)** - Get day's high price
- **get_stock_day_low(symbol)** - Get day's low price
- **get_stock_exchange(symbol)** - Get stock exchange
- **get_stock_fifty_day_average(symbol)** - Get 50-day moving average
- **get_stock_last_price(symbol)** - Get current stock price
- **get_stock_last_volume(symbol)** - Get last trading volume
- **get_stock_market_cap(symbol)** - Get market capitalization
- **get_stock_history(symbol, period)** - Get historical stock data
- **get_stock_income_statement(symbol)** - Get income statement data
- **get_stock_options_chain(symbol)** - Get options chain data

### Web Search Tools

- **google_search(query)** - Search the web for financial information and news

## Error Handling

The API uses standard HTTP status codes:

- **200** - Success
- **400** - Bad Request
- **401** - Unauthorized
- **403** - Forbidden
- **404** - Not Found
- **500** - Internal Server Error

Error responses include a message field:

```json
{
  "detail": "Error message description"
}
```

## Rate Limiting

Currently, no rate limiting is implemented, but it may be added in future versions.

## Interactive API Documentation

You can explore the API interactively at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Examples

### Chat with Stock Query

```bash
curl -X POST "http://localhost:8000/api/chat/stream" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "chat-1",
    "messages": [
      {
        "role": "user",
        "content": "What is the current price of Apple stock?",
        "parts": [
          {
            "type": "text",
            "text": "What is the current price of Apple stock?"
          }
        ]
      }
    ],
    "trigger": "user"
  }'
```

### Register New User

```bash
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "secure_password123"
  }'
```

### Login User

```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "password": "secure_password123"
  }'
```

## Stock Symbol Format

When querying stocks, use the appropriate symbol format:

- **US Stocks**: `AAPL`, `GOOGL`, `MSFT`
- **Indian Stocks**: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- **Other Markets**: Check Yahoo Finance for the correct symbol format

## Data Sources

The API integrates with:
- **Yahoo Finance** - For stock data and financial metrics
- **Google Search** - For real-time market news and research

## Websocket Support

Currently, the API uses HTTP streaming for real-time chat responses. Websocket support may be added in future versions.

---

For more information, visit the [installation guide](README.md) or check the interactive API documentation at http://localhost:8000/docs.
