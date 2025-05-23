# Stage 1: Build the application
FROM node:18-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libsndfile1-dev \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Node.js dependency files
COPY package*.json ./
RUN npm ci

# Copy Python requirements
COPY maxim/requirements.txt ./maxim/

# Install system dependencies and NLTK
RUN apt-get update && apt-get install -y python3-pip python3-venv nano && \
    python3 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --no-cache-dir -r maxim/requirements.txt --break-system-packages && \
    python3 -c "import nltk; nltk.download('punkt_tab', download_dir='/app/nltk_data')"

# Copy the entire application
COPY . .

# Create and permission the embeddings directory
RUN mkdir -p /app/maxim/maxim/models/embeddings && \
    chown -R 999:999 /app/maxim/maxim/models/embeddings && \
    chmod -R 775 /app/maxim/maxim/models/embeddings

# Build TypeScript code
RUN npm run build

# Stage 2: Create the production image
FROM node:18-slim

# Install Python runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create uploads directory and set permissions
RUN mkdir -p /app/uploads && chmod -R 777 /app/uploads

# Copy the virtual environment from the builder stage
COPY --from=builder /app/venv ./venv

# Copy built artifacts and required directories
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/maxim ./maxim
RUN chown -R 999:999 /app/maxim && chmod -R 775 /app/maxim
COPY --from=builder /app/maxim/models ./maxim/models
COPY --from=builder /app/nltk_data ./nltk_data

# Set environment variables for Python and NLTK
ENV PATH="/app/venv/bin:$PATH"
ENV NLTK_DATA=/app/nltk_data

# Install only production Node.js dependencies
RUN npm ci --omit=dev --no-audit --no-fund

# Create a non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
USER appuser

# Expose the application port
EXPOSE 5006

# Start the application
CMD ["node", "dist/index.js"]