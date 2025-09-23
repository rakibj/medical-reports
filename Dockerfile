# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# Install system deps (needed by pdf2image, pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libmagic1 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy dependency manifests
COPY pyproject.toml uv.lock* ./

# Install deps
RUN uv sync --no-dev --frozen

# Copy source
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8080

# Env for cloud platforms (they usually inject $PORT)
# harmless default for local
ENV PORT=8080  

# ✅ Use shell form so ${PORT} is expanded at runtime
CMD ["/bin/sh", "-c", "uv run uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
