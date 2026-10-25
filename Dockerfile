# Production Dockerfile for ClipForge Video Clipping Studio
FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501 \
    HOME=/home/clipforge

# Install system dependencies: FFmpeg and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (UID 1000 required for Hugging Face Spaces & security)
RUN useradd -m -u 1000 -s /bin/bash clipforge

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure appropriate ownership and permissions
RUN chown -R clipforge:clipforge /app && \
    mkdir -p /app/temp /app/exports && \
    chown -R clipforge:clipforge /app/temp /app/exports

# Switch to non-root user
USER clipforge

# Expose Streamlit web port
EXPOSE 8501

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch application
CMD ["streamlit", "run", "app.py"]
