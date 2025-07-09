# Build stage
FROM oven/bun:1-alpine AS builder

# Install build tools for Alpine
RUN apk add --no-cache python3 make g++

WORKDIR /app

# Install dependencies
COPY package.json bun.lock ./
RUN bun install

# Copy the rest of the application
COPY . .

# Build the application
RUN bun run build

# Debug: List what was built
RUN ls -la .next/
RUN ls -la .next/standalone/

# Production stage
FROM oven/bun:1-alpine AS runner

WORKDIR /app

# Copy the standalone application
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/static ./.next/static

# The standalone output already includes package.json, so we don't need to copy it again
# Start the standalone server
CMD ["bun", "server.js"]