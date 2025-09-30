# ENBEL Climate-Health Analysis Docker Image
# ==========================================
# Multi-stage build for optimized production image

# Build stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.10-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas0 \
    liblapack3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r enbel && useradd -r -g enbel -d /app -s /bin/bash enbel

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=enbel:enbel src/ src/
COPY --chown=enbel:enbel configs/ configs/
COPY --chown=enbel:enbel scripts/ scripts/
COPY --chown=enbel:enbel tests/ tests/
COPY --chown=enbel:enbel docker/entrypoint.sh /entrypoint.sh

# Create directories for data and outputs
RUN mkdir -p data results models figures logs cache && \
    chown -R enbel:enbel /app && \
    chmod +x /entrypoint.sh

# Switch to non-root user
USER enbel

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import enbel_pp; print('ENBEL package healthy')" || exit 1

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "-m", "enbel_pp.pipeline"]

# Metadata
LABEL maintainer="ENBEL Project Team" \
      version="1.0.0" \
      description="ENBEL Climate-Health Analysis Pipeline" \
      org.opencontainers.image.source="https://github.com/enbel/climate-health-analysis" \
      org.opencontainers.image.documentation="https://enbel.github.io/climate-health-analysis" \
      org.opencontainers.image.licenses="MIT"