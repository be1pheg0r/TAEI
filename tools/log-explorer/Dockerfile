FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/var/spool/log-explorer
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends libarchive-tools && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir -r requirements.txt && useradd --uid 10001 --create-home explorer && mkdir -p /var/spool/log-explorer && chown explorer:explorer /var/spool/log-explorer
COPY backend/ ./
COPY --from=frontend /build/dist /app/frontend/dist
USER explorer
EXPOSE 8091
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8091/api/health', timeout=3)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1", "--no-access-log", "--limit-concurrency", "24", "--timeout-keep-alive", "5"]
