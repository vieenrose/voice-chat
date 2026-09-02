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
# Sampling seed for llama-server. Default -1 == llama.cpp's own default (random per
# process). Set LLM_SEED=4711 to make a benchmark run comparable to the previous one:
# tool-call routing at temperature 0.7 flips between 4/7 and 5/7 across runs, so without
# a seed "did my change help?" is unanswerable at this sample size. It changes nothing
# unless you ask it to.
LLM_SEED = int(os.getenv("LLM_SEED", "-1"))

# id -> {label, path, alias}. `label` is what the UI shows; `alias` is what
# llama-server reports in /v1/models and what must match LingStreaming's
# `model_name` (sent as the "model" field of every chat completion request).
# 0.8B is deliberately absent. Measured on this box, decode runs 236 tok/s at 0.8B
# against 172 tok/s at 2B — 0.97 s vs 1.31 s for a 220-token answer. That 0.34 s is
# spent inside a turn where the user already waits 1.0-3.6 s for first audio (search
# round-trip and TTS dominate), so it buys nothing anyone can hear. What it costs is
# reliability: every 0.8B build tested narrated its own planning, echoed the framework's
# tool template, quoted the prompt back, or repeated itself — each needing another
# filter. Unquantized f16 scored the same as q8, so this is capacity, not quantization.
MODEL_REGISTRY: dict[str, dict] = {
    # The only model. Chosen on perplexity over 194 kB of non-repeating zh-TW Wikipedia
    # prose, a deterministic measure of how far quantization moved the model -- unlike the
    # four-prompt UI matrix, whose spread (4-6 of 8 across builds differing only in
    # throughput) is noise and cannot rank anything:
    #
    #   2B  UD-Q8_K_XL 16.738   UD-Q4_K_XL 17.080   Q4_K_M 17.091
    #   4B  UD-Q8_K_XL 13.027   UD-Q4_K_XL 13.173
    #
    # Q8 is the most faithful at both sizes (~2% lower). The 4B is the reference the
    # harness and the speech-to-speech pipeline are validated against.
    #
    # MTP is ON: the weights carry NextN layers, so the model drafts its own next tokens
    # and verifies them -- accepted drafts are the tokens that would have been produced
    # anyway, which is why it recovers throughput without touching precision. Measured
    # +20% on 4B with byte-identical greedy output. See _mtp_args(); LLM_MTP=0 disables.
    #
    # Dropped, all measured on this box:
    #   Qwen3.5 2B Q8   -- 13.027 vs 16.738 perplexity; the 4B is simply better and the
    #                      VRAM is available.
    #   Bonsai 8B ternary -- 8.2B params in 2.18 GB is real compression, but in
    #                      tokenizer-independent bits/byte it lost to the 2B (1.0786 vs
    #                      0.9652). Ternary pays for its size in quality. It also needed
    #                      Prism's llama.cpp fork, a second binary to keep working.
    #   Ling 3.0 tiny   -- same score as 2B, slower (148 vs 172 tok/s), more VRAM, and
    #                      drifted into Simplified Chinese, which this demo must not do.
    "qwen3.5-4b-q8": {
        "label": "Qwen3.5 4B · Q8_K_XL",
        "path": os.getenv("LLM_PATH_4B", "/home/user/llms/mtp/Qwen3.5-4B-UD-Q8_K_XL.gguf"),
        "alias": "qwen3.5-4b",
        "mtp": True,
    },
}


DEFAULT_MODEL_ID = os.getenv("LLM_DEFAULT_MODEL_ID", "qwen3.5-4b-q8")


def _alias_to_model_id(alias: str) -> Optional[str]:
    for mid, info in MODEL_REGISTRY.items():
        if info["alias"] == alias:
            return mid
    return None


_FLAG_SUPPORT: dict[tuple[str, str], bool] = {}


def _supports_llama_flag(bin_path: str, flag: str) -> bool:
    """Does this llama-server binary advertise `flag`? Probed once per (binary, flag).

    Deliberately probed instead of version-compared: an unrecognised argument makes
    llama-server exit at startup, so a wrong guess costs a dead LLM, not a missing
    feature. If --help cannot be run, we assume the flag is absent (degrade, don't break).
    """
    key = (bin_path, flag)
    if key not in _FLAG_SUPPORT:
        ok = False
        try:
            r = subprocess.run([bin_path, "--help"], capture_output=True, text=True, timeout=15)
            ok = flag in (r.stdout or "") or flag in (r.stderr or "")
        except Exception as e:
            logger.warning(f"could not probe {bin_path} for {flag}: {e!r}")
        _FLAG_SUPPORT[key] = ok
    return _FLAG_SUPPORT[key]


class LLMServerManager:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.current_model_id: Optional[str] = None
        self.current_alias: Optional[str] = None
        self.switching = False
        self._lock = asyncio.Lock()

    def _reasoning_args(self, server_bin: str = LLAMA_SERVER_BIN) -> list[str]:
        """`--reasoning-format deepseek` moves the model's thinking pass out of
        `message.content` and into `message.reasoning_content`.

        Default is `auto`, which for this template leaves the deliberation unparsed in the
        content body — and the agent path speaks `content`, so users were hearing
        "…但根据规则，必须调用工具。所以步骤应该是先调用 get_current_datetime…" as the answer to
        "What time is it right now?". The thinking pass itself is worth keeping (tool
        routing measures 81 % with it vs 71 % without, and the 71 % variant fabricates
        weather and news), so the fix is server-side separation rather than a text filter
        or disabling thinking. Older llama.cpp builds reject unknown arguments outright —
        which would stop the server from starting at all — so the flag is probed, not
        assumed. Set LLM_REASONING_FORMAT=none to opt out.
        """
        fmt = os.getenv("LLM_REASONING_FORMAT", "deepseek").strip()
        if not fmt or fmt == "none":
            return []
        if not _supports_llama_flag(server_bin, "--reasoning-format"):
            logger.warning("this llama-server build has no --reasoning-format; thinking text will stay "
                           "in message.content (LLM_REASONING_FORMAT to override)")
            return []
        return ["--reasoning-format", fmt]

    def _mtp_args(self, info: dict, server_bin: str = LLAMA_SERVER_BIN) -> list[str]:
        """Self-speculative decoding for weights that carry MTP (NextN) layers.

        The model drafts its own next few tokens and the target model verifies them, so
        accepted drafts are exactly the tokens that would have been produced anyway —
        this is a throughput change, not a quality one. Confirmed rather than assumed:
        at temperature 0 the same UD-Q4_K_XL weights emit byte-identical text with and
        without MTP, while measuring 84.6 -> 101.9 tok/s on 4B (+20%) and 167 -> 174 on
        2B. The bigger the model, the more the drafting pays for itself.

        Probed like --reasoning-format above, because a build without it would reject
        the argument and refuse to start at all. Set LLM_MTP=0 to opt out.
        """
        if not info.get("mtp"):
            return []
        if os.getenv("LLM_MTP", "1").strip() in ("0", "false", "no"):
            return []
        if not _supports_llama_flag(server_bin, "--spec-type"):
            logger.warning("this llama-server build has no --spec-type; MTP weights will run without "
                           "speculative decoding (correct, just slower)")
            return []
        return ["--spec-type", "draft-mtp", "--spec-draft-n-max", os.getenv("LLM_MTP_DRAFT_N", "3")]

    def _server_bin(self, info: dict) -> str:
        """Which llama-server serves this entry.

        Normally the one binary in LLAMA_SERVER_BIN. An entry may name its own with
        "bin" when its weights need a build mainline cannot load — Prism\'s ternary
        Q2_0/PQ2_0 blocks, for instance, collide with upstream\'s GGML_TYPE_Q2_0 type id
        but use a different layout, so mainline rejects the file outright. Flag probing
        (_supports_llama_flag) is already keyed by binary, so a fork with a different
        flag set is handled without further special-casing.
        """
        return info.get("bin") or LLAMA_SERVER_BIN

    def _spawn(self, model_id: str) -> None:
        info = MODEL_REGISTRY[model_id]
        path = Path(info["path"])
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        server_bin = self._server_bin(info)
        cmd = [
            server_bin, "-m", str(path),
            "--host", LLM_HOST, "--port", str(LLM_PORT),
            "-c", str(LLM_CTX), "--alias", info["alias"],
            "--n-gpu-layers", "99", "--jinja",
        ]
        cmd += self._reasoning_args(server_bin)
        cmd += self._mtp_args(info, server_bin)
        if LLM_SEED != -1:
            cmd += ["--seed", str(LLM_SEED)]
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
        we can start our own in its place.

        Restricted to processes that actually look like llama-server: POST /api/model
        reaches this path, and terminating *whatever* PID happened to hold :11435
        meant an unauthenticated request could kill an unrelated service that had
        merely been assigned the same port."""
        killed_any = False
        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN and conn.pid:
                try:
                    p = psutil.Process(conn.pid)
                    name = p.name() or ""
                    if "llama-server" not in name.lower() and "llama" not in (p.cmdline()[0] or "").lower():
                        logger.error(f"LLMServerManager: :{port} is held by pid={conn.pid} ({name}) which is "
                                     "not llama-server — refusing to kill it; stop it manually and retry")
                        continue
                    logger.info(f"LLMServerManager: terminating unowned listener on :{port} pid={conn.pid} ({name})")
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
