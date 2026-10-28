# syntax=docker/dockerfile:1
# Runtime image for the API, MQTT bridge and simulator (see docker-compose.yml).
FROM python:3.12-slim

# Slow or unstable connections can cut the ~200 MB torch wheel short, which
# pip reports as a hash mismatch: wait longer, retry, and resume downloads.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 PIP_RESUME_RETRIES=10
WORKDIR /app

# CPU-only torch keeps the image small; use the CUDA index for GPU hosts.
# The cache mount keeps downloaded wheels between builds, outside the image.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

COPY src/ src/
COPY config/ config/

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
