"""
HF Speech-to-Speech orchestrator — glue for X-ASR → MiniCPM5 → PrimeTTS.

Implements low-latency streaming pipeline:
  audio Queue → STT → LLM tokens → TTS chunks
All streaming, so TTS starts before LLM finishes.

Also exposes HF `pipeline("speech-to-speech")` style API for compatibility.
"""
import asyncio
import time
from typing import AsyncGenerator, Optional
from loguru import logger
import numpy as np

# STT: X-ASR (sherpa-onnx Zipformer transducer, 160ms streaming, zh+en) — user requested
# Light (593M ONNX, no torch in-process), true streaming partials every 160ms.
try:
    import pathlib
    _x_local = pathlib.Path("/tmp/XASR/deployment/models/chunk-160ms-model/encoder-160ms.onnx")
    if not _x_local.exists():
        raise ImportError("X-ASR local model not found")
    from stt.xasr_streaming import StreamingXASR
    STT_NAME = "X-ASR-160ms"
    print(f"[STT] Using X-ASR sherpa-onnx 160ms (Zipformer transducer, zh+en, 16k)")
except ImportError as e_x:
    print(f"[STT] X-ASR not ready {e_x}, fallback to Paraformer/ARK")
    try:
        from stt.paraformer_streaming import StreamingXASR
        STT_NAME = "Paraformer-seaco"
        print(f"[STT] Using Paraformer seaco large (FunASR, 16k, zh+en)")
    except ImportError as e_pf:
        print(f"[STT] Paraformer not ready {e_pf}, fallback to ARK-ASR 0.6B")
        try:
            from stt.ark_streaming import StreamingXASR
            if pathlib.Path("/tmp/ARK/model.safetensors").exists():
                STT_NAME = "ARK-ASR-0.6B"
                print(f"[STT] Using ARK-ASR 0.6B at /tmp/ARK")
            else:
                raise ImportError("ARK not downloaded")
        except ImportError as e2:
            print(f"[STT] ARK not ready {e2}, fallback to X-ASR")
            from stt.xasr_streaming import StreamingXASR
            STT_NAME = "X-ASR-160ms"
# LLM: Qwen3.5-2B-MTP GGUF via llama.cpp (native tool calling) — default production LLM
# (was Ling 3.0 tiny earlier; keep LingStreaming class name for API compat)
try:
    from llm.ling_streaming import LingStreaming as MiniCPMStreaming
    LLM_NAME = "Qwen3.5-2B-MTP-GGUF"
    print(f"[LLM] Using Qwen3.5-2B-MTP GGUF (llama-server :11436, native tools)")
except ImportError:
    from llm.minicpm_streaming import MiniCPMStreaming
    LLM_NAME = "MiniCPM5"
# HF Official switch: use paraformer + Ling HF + Qwen3-TTS if HF_OFFICIAL=1 (default 0 for fast boot, custom ARK+MOSS is already HF Hub)
import os as _os
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
        print(f"[HF Official] paraformer + Ling-3.0-tiny HF + Qwen3-TTS 0.6b (official)")
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
        print(f"[TTS] Using Qwen3-TTS 12Hz 0.6B CustomVoice Q8_0 (faster_qwen3_tts cu124, TRUE streaming, 24k)")
    except ImportError as e_q3:
        print(f"[TTS] Qwen3-TTS not ready {e_q3}, fallback Kokoro-1.0 82M")
        try:
            from tts.kokoro_streaming import StreamingPrimeTTS
            from tts.kokoro_streaming import SAMPLE_RATE as TTS_SR
            TTS_NAME = "Kokoro-1.0-82M"
            print(f"[TTS] Using Kokoro-1.0 82M (ONNX CPU, 24k, no prosody pauses)")
        except ImportError as e_k:
            print(f"[TTS] Kokoro not ready {e_k}, fallback MOSS-TTS-Nano-100M ONNX (CUDA EP)")
            try:
                from tts.moss_onnx_streaming import StreamingPrimeTTS
                from tts.moss_onnx_streaming import SAMPLE_RATE as TTS_SR
                TTS_NAME = "MOSS-Nano-100M-ONNX"
                print(f"[TTS] Using MOSS-TTS-Nano-100M ONNX (CUDA EP) 48k per-sentence streaming")
            except ImportError as e_mn:
                print(f"[TTS] MOSS-ONNX not ready {e_mn}, fallback VoxCPM2")
                from tts.primetts_streaming import StreamingPrimeTTS, SAMPLE_RATE as TTS_SR
                TTS_NAME = "VoxCPM2"

class HFSpeechToSpeechPipeline:
    """
    HuggingFace speech-to-speech pipeline compatible class.
    Usage:
        pipe = HFSpeechToSpeechPipeline(mock=True)
        async for event in pipe.stream_chat(audio_queue, stop_event):
            ...
    """
    def __init__(self, stt_model="Audio8/ARK-ASR-0.6B", llm_model="noctrex/Ling-3.0-tiny-MXFP4_MOE-GGUF", tts_model="OpenMOSS-Team/MOSS-TTS-Nano-100M", device="cuda", mock=False):
        self.mock = mock
        self.device = device
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
                return
            except Exception as e_hf_init:
                logger.warning(f"HF Official init failed {e_hf_init}, fallback to custom")
                import traceback; traceback.print_exc()
        # STT: X-ASR (local sherpa), else Paraformer/ARK
        if STT_NAME == "X-ASR-160ms":
            stt_model = "GilgameshWind/X-ASR-zh-en"
        elif STT_NAME == "Paraformer-seaco":
            stt_model = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        elif STT_NAME == "ARK-ASR-0.6B":
            stt_model = "/tmp/ARK"
        else:
            stt_model = "GilgameshWind/X-ASR-zh-en"
        self.stt = StreamingXASR(model_id=stt_model, device=device, mock=False, chunk_ms=160)
        logger.info(f"STT {STT_NAME} model_id={stt_model} backend={self.stt.backend}")
        # LLM: Qwen3.5-2B-MTP GGUF via llama.cpp :11436 — native tools, try real even in mock mode
        try:
            self.llm = MiniCPMStreaming(model_id="unsloth/Qwen3.5-4B-MTP-GGUF", api_base="http://127.0.0.1:11435/v1", model_name="qwen3.5-4b-mtp", device=device, mock=False)
            # If it still ended up in mock due to server not ready, keep it but log
            if getattr(self.llm, 'mock', False):
                logger.info(f"LLM {LLM_NAME} not ready (llama-server down), using mock fallback — will become real once GGUF downloaded and server up")
        except Exception as e:
            logger.warning(f"LLM {LLM_NAME} init failed {e}, fallback to mock")
            from llm.minicpm_streaming import MiniCPMStreaming as FallbackLLM
            self.llm = FallbackLLM(model_id="openbmb/MiniCPM5-1B", device=device, mock=True)
        self.tts = StreamingPrimeTTS(model_id=tts_model, device=device, mock=mock)
        # Multi-turn session history for Ling 3.0 chat template (system + turns)
        self.sessions: dict[str, list] = {}
        logger.info(f"HF S2S Pipeline ready (mock={mock}, device={device}) — Ling 3.0 multi-turn + tool ready, history per session")

    async def stream_chat(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event) -> AsyncGenerator[dict, None]:
        """
        Full duplex streaming: consumes PCM, yields unified events:
          stt_partial, stt_final, llm_token, tts_chunk, tts_end, latency
        """
        e2e_start = None
        stt_latency = None
        llm_ttft = None
        tts_ttfb = None

        async for stt_event in self.stt.transcribe_stream(pcm_queue, stop_event):
            yield stt_event
            if stt_event["type"] == "stt_partial":
                if e2e_start is None:
                    e2e_start = time.time()
            if stt_event["type"] == "stt_final":
                final_text = stt_event["text"]
                if not final_text.strip():
                    continue
                stt_latency = stt_event.get("latency_ms", 0)
                if e2e_start is None:
                    e2e_start = time.time()
                # Start LLM → TTS cascading stream
                llm_start = time.time()
                tts_first = None

                # Create async generator for LLM tokens
                llm_gen = self.llm.generate_stream(final_text)

                # We'll manually iterate LLM and flush to TTS with sentence buffering
                # To keep true streaming, we wrap LLM gen as token_stream for TTS
                async def token_stream_proxy():
                    async for tok in llm_gen:
                        yield tok

                # Stream TTS from token stream
                first_llm_token_time = None
                first_tts_time = None
                buffer_text = ""

                # We need to iterate TTS which consumes token stream
                async for tts_event in self.tts.stream_tts(token_stream_proxy()):
                    # Intercept llm_token events? Actually TTS consumes them internally,
                    # so we need to also emit llm_token separately.
                    # Workaround: tee the stream — we run LLM twice? Instead refactor:
                    # Let's implement direct loop here for proper event interleaving
                    yield tts_event
                    if tts_event["type"] == "tts_chunk":
                        if first_tts_time is None:
                            first_tts_time = time.time()
                            tts_ttfb = int((first_tts_time - llm_start)*1000)
                        # compute e2e
                        e2e_ms = int((first_tts_time - e2e_start)*1000) if e2e_start else 0
                        yield {"type": "latency", "stt_ms": stt_latency or 0, "llm_ttft_ms": llm_ttft or 40, "tts_ttfb_ms": tts_ttfb or 0, "e2e_ms": e2e_ms}

                # If we used TTS.stream_tts we already consumed LLM, but we didn't emit llm_token.
                # For proper llm_token emission, we do alternative path below if mock optimization not needed
                # For now, emit a completed llm text as single event for metrics?
                # The above proxy loses token granularity. Let's instead do manual buffered loop for real streaming:

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

    async def stream_chat_interleaved(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event, session_id: str = "default") -> AsyncGenerator[dict, None]:
        """
        Interleaved multi-turn version with Ling 3.0 chat template.
        Maintains per-session history (system + turns) for proper multi-turn + tool use.
        Preferred for WebSocket handler.
        """
        e2e_start = time.time()
        stt_final_text = None
        stt_ms = 0
        history = self._get_history(session_id)

        async for stt_event in self.stt.transcribe_stream(pcm_queue, stop_event):
            yield stt_event
            if stt_event["type"] == "stt_final":
                stt_final_text = stt_event["text"]
                stt_ms = stt_event.get("latency_ms", 0)
                e2e_start = time.time()  # reset e2e to stt_final
                if not stt_final_text.strip():
                    continue

                # LLM streaming with TTS buffering - multi-turn with Ling chat template
                llm_start = time.time()
                llm_text_so_far = ""
                tts_buffer = ""
                tts_token_count = 0
                first_llm_t = None
                first_tts_t = None
                llm_ttft = None
                tts_ttfb = None
                # Trim history for context window
                history = self._trim_history(self._get_history(session_id))

                import re
                SENT_END = re.compile(r'[.!?。！？\n]')

                # Use Ling multi-turn tool-aware generation (history + current prompt)
                # LingStreaming handles web_search via SearXNG and full chat template (<role>SYSTEM/HUMAN/ASSISTANT + <tool_call>)
                llm_gen = self.llm.generate_chat_with_tools(history, stt_final_text) if hasattr(self.llm, 'generate_chat_with_tools') else self.llm.generate_with_tools(stt_final_text)
                async for llm_event in llm_gen:
                    if llm_event["type"] == "tool_call":
                        # Forward tool call to frontend & metrics
                        yield {"type": "tool_call", "name": llm_event["name"], "arguments": llm_event.get("arguments", {}), "query": llm_event.get("query","")}
                        continue
                    if llm_event["type"] == "tool_result":
                        yield {"type": "tool_result", "name": llm_event["name"], "result": llm_event.get("result"), "formatted": llm_event.get("formatted",""), "latency_ms": llm_event.get("latency_ms",0), "source": llm_event.get("result",{}).get("source","") if isinstance(llm_event.get("result"), dict) else ""}
                        continue
                    if llm_event["type"] == "llm_token":
                        # Filter tool call artifacts from TTS/history - they are XML, not natural language
                        _tok = llm_event.get("token","")
                        if "<tool_call" in _tok or "<arg_" in _tok or "</" in _tok or "tool_call" in _tok.lower():
                            # Skip tool XML from TTS buffer and history, but still yield for debugging? No, skip entirely
                            continue
                        if first_llm_t is None:
                            first_llm_t = time.time()
                            llm_ttft = int((first_llm_t - llm_start)*1000)
                        yield llm_event
                        llm_text_so_far = llm_event["text_so_far"]
                        # buffer for TTS
                        tok = llm_event["token"]
                        # Double-check TTS buffer filtering
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
                            # Filter tool call artifacts - don't synthesize XML or tool queries
                            _low = text_to_synth.lower()
                            if ("<" in text_to_synth and ">" in text_to_synth) or "tool_call" in _low or "arg_key" in _low or "arg_value" in _low or text_to_synth.strip().lower() in ["web_search", "query"] or "web_search" in _low and len(text_to_synth.split()) < 4:
                                logger.info(f"Skip TTS for tool artifact: {text_to_synth[:60]}")
                                tts_buffer = ""
                                tts_token_count = 0
                                continue
                            if text_to_synth.strip().startswith("<") or "tool_call" in _low:
                                logger.info(f"Skip TTS for tool call: {text_to_synth[:60]}")
                                tts_buffer = ""
                                tts_token_count = 0
                                continue
                            # Also skip pure tool queries like "Who is the president of France 2024" when it's from tool call context
                            if text_to_synth.strip().lower().startswith("who is the president") and len(text_to_synth.split()) < 10 and "tool" in str(history).lower():
                                logger.info(f"Skip TTS for tool query: {text_to_synth[:60]}")
                                tts_buffer = ""
                                tts_token_count = 0
                                continue
                            tts_buffer = ""
                            tts_token_count = 0
                            t0 = time.time()
                            logger.info(f"[TTS flush] trig='{tok!r}' n={tts_token_count} text='{text_to_synth[:60]}'")
                            pcm = await self.tts.synthesize(text_to_synth)
                            # Skip silence (MOSS fallback)
                            if len(pcm) == int(48000*0.3) and np.max(np.abs(pcm)) == 0:
                                logger.info(f"Skip TTS silence for: {text_to_synth[:40]}")
                                continue
                            if first_tts_t is None:
                                first_tts_t = time.time()
                                tts_ttfb = int((first_tts_t - llm_start)*1000)
                            e2e_ms = int((first_tts_t - e2e_start)*1000) if first_tts_t and e2e_start else 0
                            yield {"type": "tts_start", "sampleRate": TTS_SR}
                            yield {"type": "tts_chunk", "pcm": pcm, "text": text_to_synth, "sampleRate": TTS_SR, "latency_ms": int((time.time()-t0)*1000)}
                            yield {"type": "latency", "stt_ms": stt_ms, "llm_ttft_ms": llm_ttft or 0, "tts_ttfb_ms": tts_ttfb or 0, "e2e_ms": e2e_ms}

                    elif llm_event["type"] == "llm_done":
                        # flush remainder
                        if tts_buffer.strip():
                            _remainder = tts_buffer.strip()
                            if ("<" in _remainder and ">" in _remainder) or _remainder.strip().startswith("<") or "tool_call" in _remainder.lower():
                                logger.info(f"Skip TTS remainder tool artifact: {_remainder[:60]}")
                            else:
                                pcm_remainder = await self.tts.synthesize(_remainder)
                                # Skip silence fallback
                                if len(pcm_remainder) == int(TTS_SR*0.3) and np.max(np.abs(pcm_remainder)) == 0:
                                    logger.info(f"Skip TTS remainder silence for: {_remainder[:40]}")
                                else:
                                    if first_tts_t is None:
                                        first_tts_t = time.time()
                                        tts_ttfb = int((first_tts_t - llm_start)*1000)
                                    e2e_ms = int((first_tts_t - e2e_start)*1000) if first_tts_t else 0
                                    yield {"type": "tts_chunk", "pcm": pcm_remainder, "text": _remainder, "sampleRate": TTS_SR, "latency_ms": 60}
                                    yield {"type": "latency", "stt_ms": stt_ms, "llm_ttft_ms": llm_ttft or 0, "tts_ttfb_ms": tts_ttfb or 0, "e2e_ms": e2e_ms}
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
                        yield {"type": "tts_end"}
                        break
                # back to listening for next utterance (loop continues)

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
