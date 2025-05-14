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
    && rm -rf /var/lib/apt/lists/*

# Install NLTK and download punkt_tab
RUN pip install nltk --break-system-packages && python3 -c "import nltk; nltk.download('punkt_tab', download_dir='/root/nltk_data')"

# Set working directory
WORKDIR /app

# Copy Node.js dependency files
COPY package*.json ./
RUN npm ci

# Copy Python requirements and set up virtual environment
COPY maxim/requirements.txt ./maxim/
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip --break-system-packages && pip install --no-cache-dir -r maxim/requirements.txt --break-system-packages

# Copy the entire application
COPY . .

# Build TypeScript code
RUN npm run build

# Stage 2: Production image
FROM node:18-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment
COPY --from=builder /app/venv ./venv

# Copy built artifacts and required directories
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/maxim ./maxim
COPY --from=builder /app/maxim/models ./maxim/models  

# Set environment variables
ENV PATH="/app/venv/bin:$PATH"
ENV NLTK_DATA=/root/nltk_data

# Install production Node.js dependencies
RUN npm ci --omit=dev --no-audit --no-fund

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
USER appuser

# Expose port
EXPOSE 5006

# Start application
CMD ["node", "dist/index.js"]