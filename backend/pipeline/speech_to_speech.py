"""
Speech-to-speech orchestrator — glue for X-ASR → LLM (Qwen-Agent) → Qwen3-TTS.

Low-latency streaming: audio Queue → STT → LLM tokens → TTS chunks, so TTS starts
before the LLM finishes.

The "HF" in HFSpeechToSpeechPipeline is historical and misleads: NOTHING here runs
through HuggingFace's speech-to-speech abstraction. This is the project's own
orchestrator (STT pump, turn_id/barge-in state machine, sentence flushing), and the
models are HF Hub *weights* served by llama.cpp / sherpa-onnx / qwentts-cpp. The
`transformers` pipeline() API appears only on fallback rungs that a healthy
deployment never reaches.

__call__() below is a one-shot convenience API. It was written to mimic a
transformers pipeline("speech-to-speech") task, but no such task exists in
transformers (v5.15 exposes automatic-speech-recognition, text-to-audio, and
audio-classification — there is no speech-to-speech). Nothing in this repo calls
__call__, and it is NOT the live path — no streaming, no tools, no barge-in — so
measuring through it measures a different system. HF_OFFICIAL mode
(pipeline/hf_official.py) is the genuine all-HF implementation and is off by
default. HuggingFace's actual voice-agent framework is the separate
`speech-to-speech` package (github.com/huggingface/speech-to-speech), which this
project does not currently use.

(The MiniCPM5/PrimeTTS names below are import-time fallbacks and aliases only.)
"""
import asyncio
import os as _os
import re
import time
from typing import AsyncGenerator, Awaitable, Callable, Optional
from loguru import logger
import numpy as np

from tts.spoken_text import normalize as _normalize_spoken_text   # stdlib-only, no engine deps
from llm.ling_streaming import SpokenGuard, retract_span as _retract_span


# Text that is tool plumbing rather than speech must never reach the speaker. One case is
# not XML: a small model "calling" a tool by printing its name and then answering
# (現在的時間是 [get_current_datetime]。). agent/qwen_harness.detect_unexecuted_tool repairs
# the answer; this guard keeps the broken draft out of the audio while the repair runs.
_TOOL_NAME_RE = re.compile(r"\[\s*(get_current_datetime|get_weather|web_search)\s*\]")
# CJK has no word separators, so `len(s.split()) < 4` is true for almost every Chinese
# sentence — the pre-existing "short fragment mentioning web_search" junk filter used to
# silence legitimate spoken Chinese ("我可以用 web_search 幫你查…"). It was written for bare
# Latin fragments like `web_search` / `query`, so it now only applies to those.
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _norm_phrase(s: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff\u3400-\u4dbf]+", " ", (s or "").lower()).strip()


def is_echo_of_prompt(candidate: str, question: str = "", tool_queries=()) -> bool:
    """True when a would-be spoken sentence is really an echo of the user's own question
    (or of a query a tool was called with) rather than an answer to it.

    Small models sometimes read the prompt back before answering. The original
    suppression for that matched the literal opening words of ONE benchmark question — a
    rule that would silently skip any answer beginning with those words and nothing else.
    Compare content instead: a short candidate whose normalized text IS the question (or a
    prefix of it), or equals a query a tool was called with this turn.
    """
    c = _norm_phrase(candidate)
    if not c or len(c.split()) > 14:
        return False
    q = _norm_phrase(question)
    if q and (c == q or (len(c.split()) >= 3 and q.startswith(c))):
        return True
    return any(c == _norm_phrase(tq) for tq in tool_queries if tq)


def _is_tool_artifact(s: str) -> bool:
    low = s.lower()
    return bool(_TOOL_NAME_RE.search(s)) or (
        ("<" in s and ">" in s)
        or "tool_call" in low or "arg_key" in low or "arg_value" in low
        or s.strip().lower() in ["web_search", "query"]
        or ("web_search" in low and len(s.split()) < 4 and not _CJK_RE.search(s)))

# STT: X-ASR (sherpa-onnx Zipformer transducer, 160ms streaming, zh+en) — user requested
# Light (593M ONNX, no torch in-process), true streaming partials every 160ms.
try:
    import pathlib
    _x_local = pathlib.Path("/tmp/XASR/deployment/models/chunk-160ms-model/encoder-160ms.onnx")
    if not _x_local.exists():
        raise ImportError("X-ASR local model not found")
    from stt.xasr_streaming import StreamingXASR
    STT_NAME = "X-ASR-160ms"
    print("[STT] Using X-ASR sherpa-onnx 160ms (Zipformer transducer, zh+en, 16k)")
except ImportError as e_x:
    print(f"[STT] X-ASR not ready {e_x}, fallback to Paraformer/ARK")
    try:
        from stt.paraformer_streaming import StreamingXASR
        STT_NAME = "Paraformer-seaco"
        print("[STT] Using Paraformer seaco large (FunASR, 16k, zh+en)")
    except ImportError as e_pf:
        print(f"[STT] Paraformer not ready {e_pf}, fallback to ARK-ASR 0.6B")
        try:
            from stt.ark_streaming import StreamingXASR
            if pathlib.Path("/tmp/ARK/model.safetensors").exists():
                STT_NAME = "ARK-ASR-0.6B"
                print("[STT] Using ARK-ASR 0.6B at /tmp/ARK")
            else:
                raise ImportError("ARK not downloaded")
        except ImportError as e2:
            print(f"[STT] ARK not ready {e2}, falling back to MOCK STT")
            # Previously the last rung re-imported X-ASR — the one that just failed —
            # so the ImportError escaped the module import and even `app.py --mock`
            # could not start without the model libraries installed.
            from stt.mock_streaming import StreamingXASR
            STT_NAME = "MOCK"
            print("[STT] Using MOCK STT adapter (energy-segmented fake transcripts)")
# LLM: Qwen3.5-0.8B-MTP GGUF via llama.cpp (native tool calling) — default production LLM
# (was Ling 3.0 tiny earlier; keep LingStreaming class name for API compat)
try:
    from llm.ling_streaming import LingStreaming as MiniCPMStreaming
    LLM_NAME = "Qwen3.5-0.8B-MTP-GGUF"
    print("[LLM] configured at runtime (llama-server, native tools)")
except ImportError:
    from llm.minicpm_streaming import MiniCPMStreaming
    LLM_NAME = "MiniCPM5"
# HF Official switch: use paraformer + Ling HF + Qwen3-TTS if HF_OFFICIAL=1 (default 0 for fast boot, custom ARK+MOSS is already HF Hub)
HF_OFFICIAL = False  # force custom ARK+MOSS for now, HF download is slow (SenseVoice 936M @ 500kB/s)
# To enable HF official: set HF_OFFICIAL=True manually or via HF_OFFICIAL=1 env and ensure models cached
if HF_OFFICIAL:
    try:
        from pipeline.hf_official import HFOfficialSTT as _HFSTT, HFOfficialLLM as _HFLLM, HFOfficialTTS as _HFTTS, SAMPLE_RATE as _HFSR
        # Use HF official names for health
        STT_NAME = "paraformer-hf"
        LLM_NAME = "ling-3.0-tiny-hf"
        TTS_NAME = "qwen3-tts-0.6b-hf"
        TTS_SR = _HFSR
        # For HFOfficialPipeline we will directly use HFOfficialPipeline class, not individual STT/LLM/TTS here
        # This block just sets names; actual pipeline will be HFOfficialPipeline
        print("[HF Official] paraformer + Ling-3.0-tiny HF + Qwen3-TTS 0.6b (official)")
    except Exception as e_hf:
        print(f"[HF Official] not ready {e_hf}, fallback to MOSS")
        HF_OFFICIAL = False
if not HF_OFFICIAL:
    # TTS default: Qwen3-TTS 0.6B CustomVoice Q8_0 — TRUE streaming via faster_qwen3_tts (qwentts-cpp cu124),
    # TTFA ~20ms, GGML CUDA. Speaks whole sentences (app.py flush), prosody pauses capped by _compress_silence.
    # Fallback 1: Kokoro-1.0 82M ONNX. Fallback 2: MOSS-TTS-Nano-100M ONNX (CUDA EP).
    try:
        from tts.qwen3_streaming import StreamingPrimeTTS
        from tts.qwen3_streaming import SAMPLE_RATE as TTS_SR
        TTS_NAME = "Qwen3-TTS-0.6b-CustomVoice-Q8"
        print("[TTS] Using Qwen3-TTS 12Hz 0.6B CustomVoice Q8_0 (faster_qwen3_tts cu124, TRUE streaming, 24k)")
    except ImportError as e_q3:
        print(f"[TTS] Qwen3-TTS not ready {e_q3}, fallback Kokoro-1.0 82M")
        try:
            from tts.kokoro_streaming import StreamingPrimeTTS
            from tts.kokoro_streaming import SAMPLE_RATE as TTS_SR
            TTS_NAME = "Kokoro-1.0-82M"
            print("[TTS] Using Kokoro-1.0 82M (ONNX CPU, 24k, no prosody pauses)")
        except ImportError as e_k:
            print(f"[TTS] Kokoro not ready {e_k}, fallback MOSS-TTS-Nano-100M ONNX (CUDA EP)")
            try:
                from tts.moss_onnx_streaming import StreamingPrimeTTS
                from tts.moss_onnx_streaming import SAMPLE_RATE as TTS_SR
                TTS_NAME = "MOSS-Nano-100M-ONNX"
                print("[TTS] Using MOSS-TTS-Nano-100M ONNX (CUDA EP) 48k per-sentence streaming")
            except ImportError as e_mn:
                print(f"[TTS] MOSS-ONNX not ready {e_mn}, falling back to MOCK TTS")
                try:
                    from tts.primetts_streaming import StreamingPrimeTTS
                    from tts.primetts_streaming import SAMPLE_RATE as TTS_SR
                    TTS_NAME = "VoxCPM2"
                except ImportError as e_vx:
                    # Terminal rung: previously VoxCPM (hard onnx/vLLM deps), so a
                    # machine without it could not boot even in --mock mode.
                    from tts.mock_streaming import StreamingPrimeTTS
                    from tts.mock_streaming import SAMPLE_RATE as TTS_SR
                    TTS_NAME = "MOCK"
                    print(f"[TTS] Using MOCK TTS adapter (tone audio; VoxCPM unavailable: {e_vx})")

def prepare_tts_text(text: str) -> tuple[str, list[str]]:
    """The one text front-end every TTS call must go through.

    Chat models write markdown; acoustic models are trained on read speech. Sending
    `**\u53f0\u98a8**` or `68%` straight to the TTS is what produced the measured 112%/65% CER on the
    mixed-script and markdown categories (see tts/spoken_text.py). Returning the applied
    rule names keeps it auditable: a log line or a report can say *why* the spoken text
    differs from the text bubble, instead of looking like the assistant said something else.

    Display text is never replaced: callers synthesize the returned string but still show
    the original to the user.
    """
    return _normalize_spoken_text(text)


class HFSpeechToSpeechPipeline:
    """
    HuggingFace speech-to-speech pipeline compatible class.
    Usage (the one entry point is the interleaved generator; the old non-interleaved
    `stream_chat` and the standalone `/ws/stt`, `/ws/tts`, `/ws/llm` debug sockets were
    removed — they were a second, untested copy of the same logic):
        pipe = HFSpeechToSpeechPipeline(mock=True)
        async for event in pipe.stream_chat_interleaved(pcm_queue, stop_event, session_id,
                                                       barge_in_event=ev, barge_in_lock=lock,
                                                       on_new_voice_turn=cancel_text_turns):
            ...
    """
    def __init__(self, stt_model="Audio8/ARK-ASR-0.6B", llm_model="noctrex/Ling-3.0-tiny-MXFP4_MOE-GGUF", tts_model="OpenMOSS-Team/MOSS-TTS-Nano-100M", device="cuda", mock=False):
        self.mock = mock
        self.device = device
        # Tracks the in-flight voice-turn response task per session, so an external
        # barge-in (button / WS message) can cancel it — see stream_chat_interleaved.
        self._voice_response_tasks: dict[str, asyncio.Task] = {}
        # Monotonic turn counter per session, shared by the voice pipeline and the
        # text_input (direct_tts) path in app.py, so a client can tag every reply with
        # a turn_id and drop stray/stale audio from an interrupted turn instead of
        # relying on a fixed "ignore window" that also swallows the start of new replies.
        self._turn_counters: dict[str, int] = {}
        # HF Official mode: use paraformer + Ling HF + Qwen3-TTS
        if HF_OFFICIAL:
            try:
                from pipeline.hf_official import HFOfficialPipeline
                # Use official pipeline directly (it handles its own STT/LLM/TTS)
                self._hf = HFOfficialPipeline(stt_model="funasr/Paraformer-large", llm_model="inclusionAI/Ling-3.0-tiny-int4", tts_model="Qwen/Qwen3-TTS-12Hz-0.6B-Base", device=device, mock=mock)
                # Expose same attributes for health
                self.stt = self._hf.stt
                self.llm = self._hf.llm
                self.tts = self._hf.tts
                self.sample_rate = self._hf.sample_rate
                self.sessions = self._hf.sessions
                # Delegate methods
                self.stream_chat_interleaved = self._hf.stream_chat_interleaved
                self._get_history = self._hf._get_history
                self._trim_history = self._hf._trim_history
                logger.info(f"HF Official S2S ready (paraformer + Ling HF + Qwen3-TTS) sr={self.sample_rate}")
                logger.info(
                    "HF_OFFICIAL mode delegates stream_chat_interleaved to HFOfficialPipeline, which "
                    "implements the same pump/turn_id/barge-in contract as this one (see hf_official.py); "
                    "self._voice_response_tasks is still owned by the delegate, so app.py's do_barge_in "
                    "cancels via barge_in_event on that path rather than via this dict."
                )
                return
            except Exception as e_hf_init:
                logger.warning(f"HF Official init failed {e_hf_init}, fallback to custom")
                import traceback
                traceback.print_exc()
        # STT: X-ASR (local sherpa), else Paraformer/ARK
        if STT_NAME == "X-ASR-160ms":
            stt_model = "GilgameshWind/X-ASR-zh-en"
        elif STT_NAME == "Paraformer-seaco":
            stt_model = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        elif STT_NAME == "ARK-ASR-0.6B":
            stt_model = "/tmp/ARK"
        else:
            stt_model = "GilgameshWind/X-ASR-zh-en"
        # Construction — not just import — is what needs a fallback here. An adapter can
        # import cleanly and then fail inside __init__ (paraformer imports funasr in its
        # constructor and raises RuntimeError), and the module-level ladder only proves
        # the import. That is why a machine without the model stack died right here, even
        # with --mock, which the mock rungs were added to make unnecessary.
        try:
            self.stt = StreamingXASR(model_id=stt_model, device=device, mock=False, chunk_ms=160)
            logger.info(f"STT {STT_NAME} model_id={stt_model} backend={self.stt.backend}")
        except Exception as e_stt_init:
            logger.warning(f"STT {STT_NAME} would not construct ({e_stt_init!r}) — falling back to the MOCK STT "
                           "adapter. /health reports models_loaded.stt='mock'; nothing is silently faked on a healthy boot.")
            from stt.mock_streaming import StreamingXASR as _MockSTT
            self.stt = _MockSTT(model_id=stt_model, device="cpu", mock=True, chunk_ms=160)
        # LLM: Qwen3.5-0.8B-MTP GGUF via llama.cpp :11436 — native tools, try real even in mock mode
        try:
            import os
            _llm_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:11435/v1")
            self.llm = MiniCPMStreaming(model_id="unsloth/Qwen3.5-2B-GGUF", api_base=_llm_base, model_name="qwen3.5-2b",
                                        device=device, mock=False, degraded_mode="mock" if mock else "error")
            # If it still ended up in mock due to server not ready, keep it but log
            if getattr(self.llm, 'mock', False):
                logger.info(f"LLM {LLM_NAME} not ready (llama-server down), using mock fallback — will become real once GGUF downloaded and server up")
        except Exception as e:
            logger.warning(f"LLM {LLM_NAME} init failed {e}, fallback to mock")
            from llm.minicpm_streaming import MiniCPMStreaming as FallbackLLM
            self.llm = FallbackLLM(model_id="openbmb/MiniCPM5-1B", device=device, mock=True)
        try:
            self.tts = StreamingPrimeTTS(model_id=tts_model, device=device, mock=mock)
        except Exception as e_tts_init:
            logger.warning(f"TTS {TTS_NAME} would not construct ({e_tts_init!r}) — falling back to the MOCK TTS "
                           "adapter (tone audio). /health reports models_loaded.tts='mock'.")
            from tts.mock_streaming import StreamingPrimeTTS as _MockTTS
            self.tts = _MockTTS(model_id=tts_model, device="cpu", mock=True)
        # Multi-turn session history for Ling 3.0 chat template (system + turns)
        self.sessions: dict[str, list] = {}
        logger.info(f"HF S2S Pipeline ready (mock={mock}, device={device}) — Ling 3.0 multi-turn + tool ready, history per session")

    def next_turn_id(self, session_id: str) -> int:
        n = self._turn_counters.get(session_id, 0) + 1
        self._turn_counters[session_id] = n
        return n

    def _get_history(self, session_id: str) -> list:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def _trim_history(self, history: list, max_turns: int = 20):
        # Keep system + last max_turns*2 messages (user+assistant), preserve tool calls
        # Ling 131k ctx, but we trim to 20 turns for low latency
        if len(history) <= 1 + max_turns*2:
            return history
        # Keep system at 0, then last N
        return [history[0]] + history[-(max_turns*2):]

    async def stream_chat_interleaved(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event, session_id: str = "default", barge_in_event: Optional[asyncio.Event] = None, barge_in_lock: Optional[asyncio.Lock] = None, on_new_voice_turn: Optional["Callable[[], Awaitable[None]]"] = None) -> AsyncGenerator[dict, None]:
        """
        Interleaved multi-turn version with Ling 3.0 chat template.
        Maintains per-session history (system + turns) for proper multi-turn + tool use.
        Preferred for WebSocket handler.

        Runs STT continuously via a background pump task instead of blocking on the
        current turn's LLM/TTS, so new speech is still recognized while a reply is
        playing. A fresh `stt_partial` (or `stt_final`) while a response is in flight
        is a voice barge-in: the in-flight response task is cancelled immediately and
        a `{"type":"barge_in","reason":"voice"}` event is yielded. `barge_in_event` is
        also checked/settable so an external trigger (button, WS message) can cancel
        the current turn the same way — see app.py's `do_barge_in`. `barge_in_lock`
        (shared with `do_barge_in`) serializes the two cancellation paths' set->clear
        sequences on `barge_in_event` so they can't interleave and let a stray chunk
        past a mid-sequence is_set() check. `on_new_voice_turn`, if given, is awaited
        right before starting each new voice turn so it can cancel a concurrently
        in-flight text_input reply (app.py's `direct_tts`) — the voice side of this
        pipeline has no visibility into that task on its own, so without this callback
        a spoken utterance and a typed message sent around the same time could each
        spin up their own uncoordinated response task talking over the same socket.
        Every response event is tagged with `turn_id` so a client can drop stray audio
        from a turn that raced its own cancellation.
        """
        if barge_in_event is None:
            barge_in_event = asyncio.Event()
        if barge_in_lock is None:
            barge_in_lock = asyncio.Lock()
        out_q: asyncio.Queue = asyncio.Queue()
        import re
        SENT_END = re.compile(r'[.!?。！？\n]')

        async def stt_pump():
            try:
                async for ev in self.stt.transcribe_stream(pcm_queue, stop_event, session_id):
                    await out_q.put(("stt", ev))
            finally:
                await out_q.put(("stt_done", None))

        async def run_turn(stt_final_text: str, stt_ms: int, turn_id: int):
            e2e_start = time.time()
            llm_start = time.time()
            llm_text_so_far = ""
            tts_buffer = ""
            tts_token_count = 0
            first_llm_t = None
            first_tts_t = None
            llm_ttft = None
            tts_ttfb = None
            first_chunk_of_turn = True
            tool_queries: list[str] = []   # what the tools were asked, for echo suppression
            spoken_guard = SpokenGuard()  # never speak the same thing twice in one turn
            turn_history = self._trim_history(self._get_history(session_id))

            async def emit(ev):
                await out_q.put(("resp", turn_id, ev))

            async def synth_and_emit(text_to_synth: str):
                nonlocal first_tts_t, tts_ttfb, first_chunk_of_turn
                spoken, applied = prepare_tts_text(text_to_synth)
                if applied:
                    logger.debug(f"tts text front-end {applied}: {text_to_synth[:50]!r} -> {spoken[:50]!r}")
                if not spoken:
                    return
                t0 = time.time()
                async for pcm_chunk in self.tts.synthesize_streaming(spoken):
                    # A barge-in means STOP consuming, not skip-and-keep-draining:
                    # `continue` here kept pulling chunks and made the TTS worker
                    # synthesize the entire rest of the sentence purely to discard it
                    # (burning the GPU that the new turn needs). Breaking runs the
                    # generator's finally, which sets its stop_flag and abandons the
                    # synthesis thread immediately.
                    if barge_in_event.is_set():
                        logger.info("run_turn: barge-in set, aborting TTS stream mid-sentence")
                        break
                    if len(pcm_chunk) == 0 or int(np.max(np.abs(pcm_chunk))) == 0:
                        # Silence-placeholder guard: peak amplitude, not a fixed length.
                        # The old `len == sr*0.3` check was an incidental length match
                        # inherited from a different TTS backend at a different sample
                        # rate; amplitude catches any backend's zero-fill fallback.
                        continue
                    if first_chunk_of_turn:
                        await emit({"type": "tts_start", "sampleRate": self.tts.sample_rate})
                        first_chunk_of_turn = False
                    if first_tts_t is None:
                        first_tts_t = time.time()
                        tts_ttfb = int((first_tts_t - llm_start)*1000)
                    e2e_ms = int((first_tts_t - e2e_start)*1000) if first_tts_t else 0
                    await emit({"type": "tts_chunk", "pcm": pcm_chunk, "text": text_to_synth, "sampleRate": self.tts.sample_rate, "latency_ms": int((time.time()-t0)*1000)})
                    await emit({"type": "latency", "stt_ms": stt_ms, "llm_ttft_ms": llm_ttft or 0, "tts_ttfb_ms": tts_ttfb or 0, "e2e_ms": e2e_ms})

            # Use Ling multi-turn tool-aware generation (history + current prompt)
            # LingStreaming handles web_search via SearXNG and full chat template (<role>SYSTEM/HUMAN/ASSISTANT + <tool_call>)
            try:
                # The zh-TW default belongs in the system prompt, not glued onto the
                # user's transcript — see the same note in app.py's text path.
                llm_gen = self.llm.generate_chat_with_tools(turn_history, stt_final_text) if hasattr(self.llm, 'generate_chat_with_tools') else self.llm.generate_with_tools(stt_final_text)
                async for llm_event in llm_gen:
                    if llm_event["type"] == "llm_reset":
                        # The harness streamed answer text and then had to take it back
                        # (an XML tool call appeared mid-stream, or a tool step followed).
                        # Drop everything buffered from that speculation so it never
                        # reaches TTS, and forward the signal so the UI clears its
                        # streaming bubble too.
                        logger.info("run_turn: llm_reset — discarding speculative text")
                        tts_buffer = ""
                        tts_token_count = 0
                        llm_text_so_far = ""
                        await emit({"type": "llm_reset"})
                        continue
                    if llm_event["type"] == "tool_call":
                        _tq = llm_event.get("query") or (llm_event.get("arguments") or {}).get("query") or ""
                        if _tq:
                            tool_queries.append(_tq)
                        await emit({"type": "tool_call", "name": llm_event["name"], "arguments": llm_event.get("arguments", {}), "query": llm_event.get("query","")})
                        continue
                    if llm_event["type"] == "tool_result":
                        await emit({"type": "tool_result", "name": llm_event["name"], "result": llm_event.get("result"), "formatted": llm_event.get("formatted",""), "latency_ms": llm_event.get("latency_ms",0), "source": llm_event.get("result",{}).get("source","") if isinstance(llm_event.get("result"), dict) else ""})
                        continue
                    if llm_event["type"] in ("llm_reasoning", "reasoning"):
                        # Never spoken — forwarded to UI's reasoning panel only
                        await emit({"type": "reasoning", "text": llm_event.get("text", ""), "delta": llm_event.get("text", "")})
                        continue
                    if llm_event["type"] == "llm_token":
                        # Filter tool call artifacts and leaked reasoning from speech
                        _tok = llm_event.get("token","")
                        try:
                            from llm.ling_streaming import _is_reasoning_text as _is_r_tok
                            if _is_r_tok(_tok):
                                await emit({"type": "reasoning", "text": _tok, "delta": _tok})
                                continue
                        except Exception:
                            pass
                        if "<tool_call" in _tok or "<arg_" in _tok or "</" in _tok or "tool_call" in _tok.lower():
                            continue
                        if first_llm_t is None:
                            first_llm_t = time.time()
                            llm_ttft = int((first_llm_t - llm_start)*1000)
                        # Accumulate locally, not from the harness's cumulative copy, so
                        # tokens filtered out here stay out (see the same note in app.py).
                        llm_text_so_far += _tok
                        await emit({**llm_event, "text_so_far": llm_text_so_far})
                        tok = llm_event["token"]
                        if "<tool" in tok.lower() or ("<" in tok and ">" in tok):
                            continue
                        tts_buffer += tok
                        tts_token_count += 1
                        should_flush = False
                        if SENT_END.search(tok):
                            should_flush = True
                        elif tts_token_count >= 300:
                            # safety cap only — never chop mid-sentence (the old >=8 + ' ,' rule fragmented
                            # every sentence into ~2-word TTS chunks -> deterministic mid-sentence pauses)
                            should_flush = True

                        if should_flush and tts_buffer.strip():
                            text_to_synth = tts_buffer.strip()
                            # Defense: if a reasoning spillover arrived as llm_token (budget truncated),
                            # re-route it to reasoning channel instead of TTS.
                            try:
                                from llm.ling_streaming import _is_reasoning_text as _is_reasoning
                                if _is_reasoning(text_to_synth):
                                    await emit({"type": "reasoning", "text": text_to_synth, "delta": text_to_synth})
                                    # Take it back out of the transcript too — a token is too
                                    # small a unit to classify, so this is only recognisable
                                    # once the sentence completed, by which point it has
                                    # already streamed to the client (see app.py:_retract).
                                    llm_text_so_far = _retract_span(llm_text_so_far, text_to_synth)
                                    await emit({"type": "llm_token", "token": "", "text_so_far": llm_text_so_far})
                                    tts_buffer = ""
                                    tts_token_count = 0
                                    continue
                            except Exception:
                                pass
                            _low = text_to_synth.lower()
                            if _is_tool_artifact(text_to_synth):
                                logger.info(f"Skip TTS for tool artifact: {text_to_synth[:60]}")
                                tts_buffer = ""
                                tts_token_count = 0
                                continue
                            if text_to_synth.strip().startswith("<") or "tool_call" in _low:
                                logger.info(f"Skip TTS for tool call: {text_to_synth[:60]}")
                                tts_buffer = ""
                                tts_token_count = 0
                                continue
                            # an echo of the caller's own question is not an answer
                            if is_echo_of_prompt(text_to_synth, stt_final_text, tool_queries):
                                logger.info(f"Skip TTS for question echo: {text_to_synth[:60]}")
                                tts_buffer = ""
                                tts_token_count = 0
                                continue
                            tts_buffer = ""
                            tts_token_count = 0
                            # Never say the same thing twice in one turn (the harness
                            # re-streams an answer from the start on each new step).
                            if not spoken_guard.should_speak(text_to_synth):
                                logger.info(f"Skip TTS duplicate sentence: {text_to_synth[:60]}")
                                continue
                            logger.info(f"[TTS flush] trig='{tok!r}' text='{text_to_synth[:60]}'")
                            await synth_and_emit(text_to_synth)

                    elif llm_event["type"] == "llm_done":
                        # flush remainder
                        if tts_buffer.strip():
                            _remainder = tts_buffer.strip()
                            try:
                                from llm.ling_streaming import _is_reasoning_text as _is_reasoning2
                                if _is_reasoning2(_remainder):
                                    await emit({"type": "reasoning", "text": _remainder, "delta": _remainder})
                                elif ("<" in _remainder and ">" in _remainder) or _remainder.strip().startswith("<") or "tool_call" in _remainder.lower():
                                    logger.info(f"Skip TTS remainder tool artifact: {_remainder[:60]}")
                                elif not spoken_guard.should_speak(_remainder):
                                    logger.info(f"Skip TTS duplicate remainder: {_remainder[:60]}")
                                else:
                                    await synth_and_emit(_remainder)
                            except Exception:
                                if ("<" in _remainder and ">" in _remainder) or _remainder.strip().startswith("<") or "tool_call" in _remainder.lower():
                                    logger.info(f"Skip TTS remainder tool artifact: {_remainder[:60]}")
                                else:
                                    await synth_and_emit(_remainder)
                        # Update multi-turn history for Ling 3.0 chat template (system + turns)
                        try:
                            from llm.ling_streaming import SYSTEM_PROMPT
                            hist = self._get_history(session_id)
                            if not hist or hist[0].get("role") != "system":
                                hist.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                            hist.append({"role": "user", "content": stt_final_text})
                            hist.append({"role": "assistant", "content": llm_text_so_far})
                            self.sessions[session_id] = self._trim_history(hist)
                        except Exception as e:
                            logger.debug(f"history update failed {e}")
                        await emit({"type": "tts_end"})
                        return
            except asyncio.CancelledError:
                raise  # real barge-in cancellation — let it propagate normally, don't treat as a backend failure
            except Exception as e:
                # The LLM/TTS backend failed independently mid-turn (e.g. a model switch via
                # POST /api/model killed the llama-server connection this generation was using)
                # — without this, the exception would vanish as an unretrieved-task-exception
                # warning (nothing else awaits this task during normal operation) and the client
                # would hang forever waiting for a tts_end that will never arrive.
                logger.warning(f"run_turn: LLM/TTS generation failed mid-turn: {e!r}")
                await emit({"type": "tts_end"})

        async def cancel_current_turn(task: asyncio.Task):
            async with barge_in_lock:
                barge_in_event.set()
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    # Not the expected cancellation outcome — e.g. the task's own
                    # connection to llama-server/TTS failed independently at the same
                    # moment (a model switch killing the old server mid-generation is
                    # one real way this happens). Swallowing it silently here would
                    # make a genuine bug indistinguishable from a normal barge-in.
                    logger.warning(f"cancel_current_turn: task raised {e!r} instead of being cleanly cancelled")
                self._voice_response_tasks.pop(session_id, None)
                barge_in_event.clear()

        response_task: Optional[asyncio.Task] = None
        current_turn_id = 0
        # Latch: a text_input reply is superseded once per spoken utterance, not
        # once per partial. Reset when the utterance ends (stt_final) below.
        superseded_text_reply = False
        stt_task = asyncio.create_task(stt_pump())
        try:
            while True:
                kind, *rest = await out_q.get()
                if kind == "stt_done":
                    break
                if kind == "resp":
                    ev_turn_id, ev = rest
                    if ev_turn_id != current_turn_id:
                        continue  # stray event from a turn that was already superseded/cancelled
                    tagged = dict(ev)
                    tagged["turn_id"] = ev_turn_id
                    yield tagged
                    continue
                (ev,) = rest  # kind == "stt"
                yield ev
                _is_speech = ev["type"] in ("stt_partial", "stt_final") and ev.get("text", "").strip()
                if _is_speech and response_task is not None and not response_task.done():
                    # User is talking again (partial) or finished a new utterance (final)
                    # while the previous reply is still playing — that supersedes it.
                    # Must check non-empty text for stt_final too: not every STT backend
                    # guards this before emitting (e.g. the whisper-fallback path's flush
                    # handler yields stt_final unconditionally), and an empty "utterance"
                    # is not real new speech — it shouldn't cut off an in-flight reply.
                    await cancel_current_turn(response_task)
                    response_task = None
                    yield {"type": "barge_in", "reason": "voice", "turn_id": current_turn_id}
                elif _is_speech and on_new_voice_turn is not None and not superseded_text_reply:
                    # Same interruption, but the reply in flight came from `text_input`,
                    # so it is not `response_task` — it lives in app.py's task set and is
                    # only reachable through on_new_voice_turn.
                    #
                    # This used to be handled exclusively under `stt_final` below, i.e.
                    # only once the user had STOPPED talking and the transcript was final.
                    # Speaking over a typed reply therefore did nothing until the whole
                    # utterance finished — which is invisible on a short answer and very
                    # obvious on a long one, where the assistant talks over the user for
                    # seconds. Voice interrupting voice reacted to the first partial;
                    # voice interrupting text now does too.
                    #
                    # Latched until the utterance ends so a stream of partials cancels
                    # once rather than re-firing per partial.
                    superseded_text_reply = True
                    await on_new_voice_turn()
                    yield {"type": "barge_in", "reason": "voice", "turn_id": current_turn_id}
                if ev["type"] == "stt_final":
                    stt_final_text = ev["text"]
                    stt_ms = ev.get("latency_ms", 0)
                    _superseded_this_utterance = superseded_text_reply
                    superseded_text_reply = False      # utterance over; re-arm the latch
                    if not stt_final_text.strip():
                        continue
                    if on_new_voice_turn is not None and not _superseded_this_utterance:
                        # Symmetric with app.py's do_barge_in (which cancels an in-flight
                        # voice turn before starting a text_input reply): a fresh spoken
                        # utterance must equally supersede an in-flight text_input reply,
                        # or the two input channels could end up running concurrent,
                        # uncoordinated response tasks that both write to the same socket.
                        await on_new_voice_turn()
                    current_turn_id = self.next_turn_id(session_id)
                    response_task = asyncio.create_task(run_turn(stt_final_text, stt_ms, current_turn_id))
                    self._voice_response_tasks[session_id] = response_task
        finally:
            stt_task.cancel()
            if response_task is not None and not response_task.done():
                response_task.cancel()
            for t in (stt_task, response_task):
                if t is None:
                    continue
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug(f"stream_chat_interleaved teardown: task raised {e!r}")
            self._voice_response_tasks.pop(session_id, None)

    # HF pipeline alias
    async def __call__(self, audio_array, sampling_rate=16000):
        """One-shot speech→speech for HF pipeline compatibility"""
        import numpy as np
        # audio_array: np.ndarray float32 -1..1 or int16
        if isinstance(audio_array, np.ndarray) and audio_array.dtype == np.int16:
            pcm_f32 = audio_array.astype(np.float32)/32768.0
        else:
            pcm_f32 = np.array(audio_array, dtype=np.float32)
        # transcribe
        text = await self.stt.transcribe_once(pcm_f32)
        # llm
        llm_out = ""
        async for e in self.llm.generate_stream(text):
            if e["type"] == "llm_token":
                llm_out = e["text_so_far"]
        # tts
        chunks = []
        async for e in self.tts.tts_from_text(llm_out):
            if e["type"] == "tts_chunk":
                chunks.append(e["pcm"])
        if chunks:
            out_pcm = np.concatenate(chunks)
        else:
            out_pcm = np.zeros(1000, dtype=np.int16)
        return {"audio": out_pcm, "sampling_rate": TTS_SR, "text": llm_out, "stt_text": text}
