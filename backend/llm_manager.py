"""
Owns the lifecycle of the single llama-server subprocess that serves the
"text LLM" role (port 11435), so the UI can let the user pick among a few
Qwen3.5 sizes and have the backend actually load the chosen one.

Only one text-LLM model is loaded at a time — VRAM headroom on a 12GB card is
tight enough (embedding + TTS + STT + one LLM already used ~4.4GB at idle)
that keeping all sizes warm simultaneously would risk an OOM crash of an
unrelated model. Switching therefore stops the current llama-server and starts
the new one: expect ~5-15s of voice-chat unavailability during a switch (the
`/api/model` endpoints in app.py surface this as an explicit "switching"
state so the frontend can show a loading indicator instead of looking broken).

The embedding llama-server (port 11434) is unrelated to this and stays
externally managed as documented in the README.
"""
import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
import psutil
from loguru import logger

LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN", "/home/user/llama.cpp/build/bin/llama-server")
LLM_HOST = "127.0.0.1"
LLM_PORT = int(os.getenv("LLM_PORT", "11435"))
LLM_CTX = int(os.getenv("LLM_CTX", "16384"))

# id -> {label, path, alias}. `label` is what the UI shows; `alias` is what
# llama-server reports in /v1/models and what must match LingStreaming's
# `model_name` (sent as the "model" field of every chat completion request).
MODEL_REGISTRY: dict[str, dict] = {
    "qwen3.5-0.8b-q8": {
        "label": "Qwen3.5 0.8B Q8_0",
        "path": os.getenv("LLM_PATH_0_8B", "/tmp/llms/Qwen3.5-0.8B-Q8_0.gguf"),
        "alias": "qwen3.5-0.8b",
    },
    "qwen3.5-2b-q4": {
        "label": "Qwen3.5 2B Q4_K_M",
        "path": os.getenv("LLM_PATH_2B", "/tmp/llms/Qwen3.5-2B-Q4_K_M.gguf"),
        "alias": "qwen3.5-2b",
    },
    "qwen3.5-4b-q4": {
        "label": "Qwen3.5 4B Q4_K_M",
        "path": os.getenv("LLM_PATH_4B", "/tmp/llms/Qwen3.5-4B-Q4_K_M.gguf"),
        "alias": "qwen3.5-4b",
    },
}
DEFAULT_MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen3.5-2b-q4")


def _alias_to_model_id(alias: str) -> Optional[str]:
    for mid, info in MODEL_REGISTRY.items():
        if info["alias"] == alias:
            return mid
    return None


class LLMServerManager:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.current_model_id: Optional[str] = None
        self.current_alias: Optional[str] = None
        self.switching = False
        self._lock = asyncio.Lock()

    def _spawn(self, model_id: str) -> None:
        info = MODEL_REGISTRY[model_id]
        path = Path(info["path"])
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        cmd = [
            LLAMA_SERVER_BIN, "-m", str(path),
            "--host", LLM_HOST, "--port", str(LLM_PORT),
            "-c", str(LLM_CTX), "--alias", info["alias"],
            "--n-gpu-layers", "99", "--jinja",
        ]
        logger.info(f"LLMServerManager: spawning {model_id} ({info['label']}): {' '.join(cmd)}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _stop_owned(self) -> None:
        if self.proc is None:
            return
        pid = self.proc.pid
        logger.info(f"LLMServerManager: stopping owned llama-server pid={pid}")
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception as e:
                logger.warning(f"LLMServerManager: failed to stop pid={pid} cleanly: {e}")
        self.proc = None

    @staticmethod
    def _kill_unowned_listener(port: int) -> None:
        """Only used the first time we take over a server we didn't spawn
        ourselves (adopted an already-running one at startup) — finds
        whatever process is bound to our configured port and terminates it so
        we can start our own in its place."""
        killed_any = False
        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN and conn.pid:
                try:
                    p = psutil.Process(conn.pid)
                    logger.info(f"LLMServerManager: terminating unowned listener on :{port} pid={conn.pid} ({p.name()})")
                    p.terminate()
                    p.wait(timeout=10)
                    killed_any = True
                except Exception as e:
                    logger.warning(f"LLMServerManager: could not stop unowned pid={conn.pid} on :{port}: {e}")
        if killed_any:
            time.sleep(0.5)

    async def _wait_ready(self, timeout: float = 60.0) -> Optional[str]:
        """Polls /v1/models until it responds; returns the alias it reports, or None on timeout."""
        deadline = time.time() + timeout
        url = f"http://{LLM_HOST}:{LLM_PORT}/v1/models"
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        data = r.json().get("data", [])
                        if data:
                            return data[0].get("id") or (data[0].get("aliases") or [None])[0]
                        return ""  # server up but reports no models — still "ready" for our purposes
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        return None

    async def ensure_started(self) -> None:
        """Call once at app startup. Adopts an already-listening server on the
        configured port (e.g. this project's traditional manual-start
        workflow) rather than spawning a duplicate that would fail to bind
        the port; the first /api/model switch afterwards then takes real
        ownership by killing that adopted process."""
        async with self._lock:
            alias = await self._wait_ready(timeout=1.0)
            if alias is not None:
                self.current_alias = alias
                self.current_model_id = _alias_to_model_id(alias)
                logger.info(f"LLMServerManager: adopted already-running llama-server on :{LLM_PORT} (alias={alias!r})")
                return
            try:
                self._spawn(DEFAULT_MODEL_ID)
            except FileNotFoundError as e:
                logger.error(f"LLMServerManager: {e} — no LLM server started, pipeline will run in degraded/mock mode")
                return
            alias = await self._wait_ready()
            if alias is not None:
                self.current_model_id = DEFAULT_MODEL_ID
                self.current_alias = MODEL_REGISTRY[DEFAULT_MODEL_ID]["alias"]
                logger.info(f"LLMServerManager: {DEFAULT_MODEL_ID} ready on :{LLM_PORT}")
            else:
                logger.error(f"LLMServerManager: {DEFAULT_MODEL_ID} failed to become ready within timeout")

    async def switch_to(self, model_id: str) -> dict:
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"unknown model_id {model_id!r}; available: {list(MODEL_REGISTRY)}")
        async with self._lock:
            if model_id == self.current_model_id and (self.proc is not None or self.current_alias == MODEL_REGISTRY[model_id]["alias"]):
                return {"status": "unchanged", "model_id": model_id, "label": MODEL_REGISTRY[model_id]["label"]}
            self.switching = True
            t0 = time.time()
            try:
                if self.proc is not None:
                    self._stop_owned()
                else:
                    self._kill_unowned_listener(LLM_PORT)
                self._spawn(model_id)
                alias = await self._wait_ready()
                if alias is None:
                    raise RuntimeError(f"{model_id} did not become ready within timeout")
                self.current_model_id = model_id
                self.current_alias = alias
                return {"status": "ok", "model_id": model_id, "label": MODEL_REGISTRY[model_id]["label"], "alias": alias, "took_s": round(time.time() - t0, 1)}
            finally:
                self.switching = False

    def status(self) -> dict:
        return {
            "current_model_id": self.current_model_id,
            "current_alias": self.current_alias,
            "switching": self.switching,
            "owned": self.proc is not None,
            "available": [
                {"id": mid, "label": info["label"], "loaded": mid == self.current_model_id, "exists": Path(info["path"]).exists()}
                for mid, info in MODEL_REGISTRY.items()
            ],
        }


llm_manager = LLMServerManager()
