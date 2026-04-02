FROM python:3.11-slim

# Build args
ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user:user . .

# Create necessary directories
RUN mkdir -p envs tasks graders tests scripts && \
    touch envs/__init__.py tasks/__init__.py graders/__init__.py

USER user

# Expose port
EXPOSE 7860

ENV PORT=7860
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "app.py"]
