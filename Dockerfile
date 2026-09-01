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

# NOTE the trailing slash: with more than one source, the destination must be a
# directory. `COPY a.txt b.txt backend/a.txt` silently ships only one of them.
COPY backend/requirements.txt backend/requirements-light.txt backend/

# INSTALL_MODE=light installs only the web/streaming/audio deps and boots the service in
# mock mode (every adapter degrades to its mock rung). That is the image CI builds and
# health-checks: it cannot catch a bad model pin, but it does catch the class of failure
# that used to be invisible — a missing import, an adapter that dies at construction, a
# boot path that quietly requires an 800 MB dependency. Full mode is the real image.
#   docker build -t voice-chat .
#   docker build -t voice-chat-smoke --build-arg INSTALL_MODE=light --target smoke .
ARG INSTALL_MODE=full
RUN if [ "$INSTALL_MODE" = "light" ]; then \
      pip install -r backend/requirements-light.txt; \
    else \
      pip install -r backend/requirements.txt; \
    fi

COPY backend/ backend/
COPY --from=ui /app/frontend/dist frontend/dist

# model dirs (override via volumes): LLM gguf is consumed by llama-server (outside),
# TTS/STT weights mounted here as read-only
VOLUME ["/models/tts", "/models/stt"]
# `llm`, not `llama`: the service in docker-compose.yml is named llm, and this default is
# what a bare `docker run --link llm` gets. docker-compose.yml overrides it anyway.
ENV LLM_API_BASE=http://llm:8080/v1 \
    TTS_MODEL_DIR=/models/tts \
    STT_MODEL_DIR=/models/stt

EXPOSE 8000
ENTRYPOINT ["python"]
CMD ["backend/app.py", "--port", "8000", "--host", "0.0.0.0"]

# `smoke` is a real stage, not a duplicated Dockerfile: same base, same apt layer, same
# COPYs, same ENTRYPOINT — only INSTALL_MODE and the boot flags differ. It cannot catch a
# bad model pin, but it does catch "the service no longer starts".
#   docker build -t voice-chat-smoke --build-arg INSTALL_MODE=light --target smoke .
#   docker run --rm -p 8010:8010 voice-chat-smoke backend/app.py --mock --host 0.0.0.0 --port 8010
FROM runtime AS smoke
CMD ["backend/app.py", "--mock", "--host", "0.0.0.0", "--port", "8000"]
