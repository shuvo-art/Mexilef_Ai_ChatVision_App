# Use Node.js 18-slim as the base image for building
FROM node:18-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install ALL dependencies (including dev dependencies) for building
RUN npm ci

# Copy Python requirements
COPY maxim/requirements.txt ./maxim/

# Install Python dependencies in the virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r maxim/requirements.txt

# Copy the rest of the application
COPY . .

# Build TypeScript code
RUN npm run build

# Create production image
FROM node:18-slim

# Install Python runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/venv ./venv

# Copy built artifacts from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/maxim ./maxim
# Removed COPY --from=builder /app/src/app/config ./src/app/config since Firebase uses env variable

# Set the PATH to use the virtual environment's Python
ENV PATH="/app/venv/bin:$PATH"

# Install only production Node.js dependencies
RUN npm ci --omit=dev --no-audit --no-fund

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
USER appuser

# Expose the port
EXPOSE 5006

# Start the application
CMD ["node", "dist/index.js"]