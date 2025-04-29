# Use Node.js 18 as the base image for building
FROM node:18-alpine AS builder

# Install build dependencies
RUN apk add --no-cache python3 py3-pip build-base libsndfile-dev

# Set working directory
WORKDIR /app

# Create a virtual environment for Python
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy package.json and package-lock.json
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --omit=dev

# Copy Python requirements
COPY maxim/requirements.txt ./maxim/

# Install Python dependencies in the virtual environment
RUN pip install --no-cache-dir -r maxim/requirements.txt

# Copy the rest of the application
COPY . .

# Build TypeScript code
RUN npm run build

# Create production image
FROM node:18-alpine

# Install Python runtime dependencies
RUN apk add --no-cache python3 libsndfile

# Set working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/venv ./venv

# Copy built artifacts from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/maxim ./maxim
COPY --from=builder /app/uploads ./uploads
COPY --from=builder /app/.env ./
COPY --from=builder /app/src/app/config ./src/app/config

# Set the PATH to use the virtual environment's Python
ENV PATH="/app/venv/bin:$PATH"

# Install only production Node.js dependencies
RUN npm ci --omit=dev --no-audit --no-fund

# Create non-root user for security
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

# Expose the port
EXPOSE 5006

# Start the application
CMD ["node", "dist/index.js", "192.168.10.198:5006"]