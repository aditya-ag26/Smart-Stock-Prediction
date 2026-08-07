# syntax=docker/dockerfile:1

# ---------- Builder stage: install Python deps into an isolated venv ----------
FROM python:3.11-slim AS builder

WORKDIR /build

COPY stockproject/predictor/requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------- Final stage: minimal runtime image ----------
FROM python:3.11-slim

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app && chown appuser:appuser /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=stockproject.settings

WORKDIR /app

COPY --chown=appuser:appuser stockproject/ /app/

RUN chmod +x /app/entrypoint.sh

USER appuser

RUN python manage.py collectstatic --noinput

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/predict/health/', timeout=10)" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
