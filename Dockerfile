# syntax=docker/dockerfile:1
# Voice-chat demo: FastAPI backend (API + WS + UI) — models mounted as volumes,
# llama-server (LLM) runs as a sibling service (see docker-compose.yml).

# ---------- Stage 1: build frontend ----------
FROM node:20-alpine AS ui
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ .
RUN npm run build

# ---------- Stage 2: backend ----------
FROM python:3.13-slim AS runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# system deps (espeak-ng phonemizer for Kokoro fallback)
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng libespeak-ng1 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/ backend/
COPY --from=ui /app/frontend/dist frontend/dist

# model dirs (override via volumes): LLM gguf is consumed by llama-server (outside),
# TTS/STT weights mounted here as read-only
VOLUME ["/models/tts", "/models/stt"]
ENV LLM_API_BASE=http://llama:8080/v1 \
    TTS_MODEL_DIR=/models/tts \
    STT_MODEL_DIR=/models/stt

EXPOSE 8000
ENTRYPOINT ["python"]
CMD ["backend/app.py", "--port", "8000", "--host", "0.0.0.0"]