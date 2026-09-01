"""
HF Official Speech-to-Speech pipeline — paraformer + Ling 3.0 tiny + Qwen3-TTS 0.6b
Uses only HF pipelines / transformers, no custom llama.cpp / sherpa / codec hacks.
- STT: funasr/Paraformer-large (or iic/SenseVoiceSmall) via funasr AutoModel
- LLM: inclusionAI/Ling-3.0-tiny via transformers pipeline text-generation
- TTS: Qwen/Qwen3-TTS-12Hz-0.6B-Base via pipeline text-to-speech
All via HF Hub, CUDA bfloat16, streaming where possible.
"""
import asyncio
import time
import re
from pathlib import Path
from typing import AsyncGenerator
import numpy as np
from loguru import logger

SAMPLE_RATE = 24000  # Qwen3-TTS 12Hz -> 24000 (12Hz * 2000)
FLUSH_TOKENS = 8
SENTENCE_END = re.compile(r'[.!?。！？\n]')

class HFOfficialSTT:
    def __init__(self, model_id: str = "funasr/Paraformer-large", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.backend = "paraformer-hf"
        self.model = None
        try:
            from funasr import AutoModel
            logger.info(f"Loading Paraformer {model_id} on {device}...")
            # funasr AutoModel handles ModelScope/HF
            self.model = AutoModel(model=model_id, device="cuda:0" if device=="cuda" else "cpu", disable_update=True)
            logger.info(f"Paraformer loaded ✓ {model_id}")
        except Exception as e:
            logger.warning(f"Paraformer load failed {e}, fallback to SenseVoiceSmall")
            try:
                from funasr import AutoModel as FAuto
                self.model = FAuto(model="iic/SenseVoiceSmall", device="cuda:0" if device=="cuda" else "cpu", disable_update=True)
                self.backend = "sensevoice-hf"
                logger.info("SenseVoiceSmall loaded ✓")
            except Exception as e2:
                logger.error(f"STT fallback failed {e2}")
                raise

    async def transcribe_once(self, pcm_f32: np.ndarray) -> str:
        # pcm_f32: float32 -1..1, 16k
        def _infer():
            # funasr expects 16k mono
            import tempfile
            import soundfile as sf
            import os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                sf.write(tf.name, pcm_f32, 16000)
                tf.flush()
                res = self.model.generate(input=tf.name, batch_size_s=300)
                os.unlink(tf.name)
                if res and len(res) > 0:
                    # res is list of dicts with "text"
                    txt = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
                    return txt.strip()
                return ""
        return await asyncio.to_thread(_infer)

    async def transcribe_stream(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event, session_id: str | None = None):
        # Simple non-streaming: accumulate until flush, then transcribe_once
        buf = []
        while not stop_event.is_set():
            try:
                item = await asyncio.wait_for(pcm_queue.get(), timeout=0.3)
                if isinstance(item, dict) and item.get("type") == "flush":
                    if buf:
                        pcm = np.concatenate(buf) if buf else np.array([], dtype=np.int16)
                        pcm_f32 = pcm.astype(np.float32) / 32768.0 if pcm.dtype == np.int16 else pcm.astype(np.float32)
                        # Resample if needed (assume 16k)
                        text = await self.transcribe_once(pcm_f32)
                        yield {"type": "stt_final", "text": text, "latency_ms": 0}
                        buf = []
                    continue
                # item is pcm int16
                buf.append(item)
                # Paraformer-through-this-path has no incremental decoding, so there is
                # deliberately NO empty stt_partial filler here: the WS contract (and the
                # barge-in detector) treats a partial as "the user is saying something",
                # and the frontend paints it as live caption text.
            except asyncio.TimeoutError:
                continue

class HFOfficialLLM:
    def __init__(self, model_id: str = "inclusionAI/Ling-3.0-tiny-int4", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.backend = "ling-3.0-tiny-hf"
        self.pipe = None
        self.tokenizer = None
        self.model = None
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            logger.info(f"Loading Ling {model_id} via transformers on {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            # Try pipeline first
            try:
                self.pipe = pipeline("text-generation", model=model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=256)
                logger.info(f"Ling pipeline loaded ✓ {model_id}")
            except Exception as e_pipe:
                logger.warning(f"Ling pipeline failed {e_pipe}, try AutoModel")
                self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
                logger.info(f"Ling AutoModel loaded ✓ {model_id}")
        except Exception as e:
            logger.error(f"Ling load failed {e}")
            raise

    async def generate_stream(self, prompt: str, max_new_tokens: int = 128) -> AsyncGenerator[dict, None]:
        # Simple non-streaming then chunk
        def _gen():
            if self.pipe:
                out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, return_full_text=False)
                return out[0]["generated_text"] if isinstance(out, list) else str(out)
            else:
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.inference_mode():
                    out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
                return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        text = await asyncio.to_thread(_gen)
        # Stream as tokens
        text_so_far = ""
        for tok in text.split(" "):
            await asyncio.sleep(0.02)
            text_so_far += (" " if text_so_far else "") + tok
            yield {"type": "llm_token", "token": " " + tok if text_so_far else tok, "text_so_far": text_so_far, "latency_ms": 20}
        yield {"type": "llm_done", "text": text_so_far}

    async def generate_chat_with_tools(self, history, prompt, max_new_tokens=128):
        # For HF official, we just do generate_stream with history flattened
        # history is list of {role, content}, prompt is current user
        full_prompt = ""
        for h in history:
            if h.get("role") == "system":
                full_prompt += f"System: {h['content']}\n"
            elif h.get("role") == "user":
                full_prompt += f"User: {h['content']}\n"
            elif h.get("role") == "assistant":
                full_prompt += f"Assistant: {h['content']}\n"
        full_prompt += f"User: {prompt}\nAssistant:"
        async for ev in self.generate_stream(full_prompt, max_new_tokens=max_new_tokens):
            yield ev

    async def generate_with_tools(self, prompt, max_new_tokens=128):
        async for ev in self.generate_chat_with_tools([], prompt, max_new_tokens):
            yield ev

class HFOfficialTTS:
    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.backend = "qwen3-tts-0.6b-hf"
        self.pipe = None
        self.sample_rate = SAMPLE_RATE
        try:
            from transformers import pipeline
            import torch
            logger.info(f"Loading Qwen3-TTS {model_id} on {device}...")
            # Qwen3-TTS needs speaker
            self.pipe = pipeline("text-to-speech", model=model_id, device="cuda:0" if device=="cuda" else "cpu", torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32)
            # Try to get sample_rate from model
            try:
                self.sample_rate = getattr(self.pipe.model.config, "sampling_rate", 24000)
            except Exception:
                pass
            logger.info(f"Qwen3-TTS loaded ✓ {model_id} sr {self.sample_rate}")
        except Exception as e:
            logger.warning(f"Qwen3-TTS pipeline failed {e}, try alternative")
            try:
                from transformers import AutoModel, AutoProcessor
                # Fallback to Qwen2.5-TTS or MMS
                fallback = "facebook/mms-tts-eng" if "qwen" in model_id.lower() else model_id
                self.pipe = pipeline("text-to-speech", model=fallback, device="cuda:0" if device=="cuda" else "cpu")
                logger.info(f"Fallback TTS {fallback} loaded")
            except Exception as e2:
                logger.error(f"TTS fallback failed {e2}")
                raise

    async def synthesize(self, text: str) -> np.ndarray:
        def _infer():
            # Qwen3-TTS expects text and speaker
            try:
                out = self.pipe(text)
                # out is dict with "audio" and "sampling_rate"
                wav = out["audio"] if isinstance(out, dict) else out
                # wav is float32 -1..1 or numpy
                import numpy as np
                wav = np.array(wav).squeeze()
                # Convert to int16
                pcm = (wav * 32767).astype(np.int16)
                return pcm
            except Exception:
                # Try with speaker param
                try:
                    out = self.pipe(text, forward_params={"speaker": "Chelsie"})
                    wav = out["audio"] if isinstance(out, dict) else out
                    wav = np.array(wav).squeeze()
                    return (wav * 32767).astype(np.int16)
                except Exception:
                    raise
        try:
            pcm = await asyncio.to_thread(_infer)
            return pcm
        except Exception as e:
            logger.error(f"Qwen TTS failed {e}")
            # Return silence 0.5s
            return np.zeros(int(self.sample_rate*0.5), dtype=np.int16)

    async def synthesize_streaming(self, text: str):
        pcm = await self.synthesize(text)
        # Yield in 0.5s chunks
        chunk = int(self.sample_rate * 0.5)
        for i in range(0, len(pcm), chunk):
            yield pcm[i:i+chunk]
            await asyncio.sleep(0.01)

    async def tts_from_text(self, text: str):
        import re
        sents = re.split(r'([.!?]+)', text)
        for i in range(0, len(sents), 2):
            sent = (sents[i] + (sents[i+1] if i+1 < len(sents) else "")).strip()
            if not sent:
                continue
            async for c in self.synthesize_streaming(sent):
                yield {"type": "tts_chunk", "pcm": c, "text": sent, "sampleRate": self.sample_rate, "latency_ms": 40}
        yield {"type": "tts_end"}

    async def stream_tts(self, token_stream):
        buf=""
        cnt=0
        async for ev in token_stream:
            if ev["type"]=="llm_token":
                buf+=ev["token"]
                cnt+=1
                should=False
                if re.search(r'[.!?。！？\n]', ev["token"]):
                    should=True
                elif cnt>=8 and buf and buf[-1] in " ,":
                    should=True
                if should and buf.strip():
                    txt=buf.strip()
                    buf=""
                    cnt=0
                    async for c in self.synthesize_streaming(txt):
                        yield {"type":"tts_chunk","pcm":c,"text":txt,"sampleRate":self.sample_rate,"latency_ms":40}
            elif ev["type"]=="llm_done":
                if buf.strip():
                    async for c in self.synthesize_streaming(buf.strip()):
                        yield {"type":"tts_chunk","pcm":c,"text":buf.strip(),"sampleRate":self.sample_rate,"latency_ms":40}
                yield {"type":"tts_end"}
                return
        if buf.strip():
            async for c in self.synthesize_streaming(buf.strip()):
                yield {"type":"tts_chunk","pcm":c,"text":buf.strip(),"sampleRate":self.sample_rate,"latency_ms":40}
        yield {"type":"tts_end"}

class HFOfficialPipeline:
    """Drop-in replacement for HFSpeechToSpeechPipeline using HF official pipelines."""
    def __init__(self, stt_model="funasr/Paraformer-large", llm_model="inclusionAI/Ling-3.0-tiny-int4", tts_model="Qwen/Qwen3-TTS-12Hz-0.6B-Base", device="cuda", mock=False):
        self.mock = mock
        self.device = device
        self.sessions = {}
        # STT
        try:
            self.stt = HFOfficialSTT(model_id=stt_model, device=device)
        except Exception as e:
            logger.warning(f"HF STT failed {e}, fallback to ARK")
            from stt.ark_streaming import StreamingXASR
            self.stt = StreamingXASR(model_id="/tmp/ARK", device=device)
        # LLM
        try:
            self.llm = HFOfficialLLM(model_id=llm_model, device=device)
        except Exception as e:
            logger.warning(f"HF LLM failed {e}, fallback to Ling GGUF")
            from llm.ling_streaming import LingStreaming
            self.llm = LingStreaming()
        # TTS
        try:
            self.tts = HFOfficialTTS(model_id=tts_model, device=device)
            self.sample_rate = self.tts.sample_rate
        except Exception as e:
            logger.warning(f"HF TTS failed {e}, fallback to MOSS")
            from tts.moss_nano_streaming import StreamingPrimeTTS
            self.tts = StreamingPrimeTTS(device=device)
            self.sample_rate = self.tts.sample_rate
        logger.info(f"HF Official S2S ready (mock={mock}) stt={self.stt.backend} llm={self.llm.backend} tts={self.tts.backend}")

    def _get_history(self, sid): 
        return self.sessions.get(sid, [])
    def _trim_history(self, h, max_turns=10):
        if len(h) <= 1+max_turns*2:
            return h
        return [h[0]] + h[-(max_turns*2):]

    def next_turn_id(self, session_id: str = "default") -> int:
        """Monotonic per-session turn id, same contract as the default pipeline's,
        so app.py can tag/drop stale audio identically on either pipeline."""
        self._turn_counters = getattr(self, "_turn_counters", {})
        self._turn_counters[session_id] = self._turn_counters.get(session_id, 0) + 1
        return self._turn_counters[session_id]

    async def stream_chat_interleaved(self, pcm_queue, stop_event, session_id="default", barge_in_event=None,
                                      barge_in_lock=None, on_new_voice_turn=None):
        """
        Barge-in / turn_id parity with the default pipeline (speech_to_speech.py).

        STT runs in a background pump so speech keeps being recognized while a reply is
        playing; each utterance gets its own response task, and every response event is
        tagged with `turn_id`. A fresh non-empty `stt_partial` (or any `stt_final`) while
        a reply is in flight cancels that task and yields
        `{"type":"barge_in","reason":"voice","turn_id":…}`; `barge_in_event` (set by
        app.py's `do_barge_in`) is checked before each synthesis and before each emitted
        chunk, and breaking out of the TTS generator runs its `finally` -> stop_flag, so
        an interrupted sentence is abandoned rather than synthesized to be thrown away.
        `on_new_voice_turn` is awaited before each new voice turn to cancel a concurrent
        text_input reply, exactly as the default pipeline does.
        """
        if barge_in_event is None:
            barge_in_event = asyncio.Event()
        if barge_in_lock is None:
            barge_in_lock = asyncio.Lock()
        out_q: asyncio.Queue = asyncio.Queue()

        async def stt_pump():
            try:
                async for ev in self.stt.transcribe_stream(pcm_queue, stop_event, session_id):
                    await out_q.put(("stt", ev))
            except Exception as e:                       # pump death must not hang the loop
                logger.warning(f"hf stt pump ended: {e!r}")
            finally:
                await out_q.put(("stt", {"type": "_pump_done"}))

        async def run_turn(txt: str, stt_ms: int, turn_id: int):
            e2e_start = time.time()
            first_llm = None
            first_tts = None
            tts_buf = ""
            cnt = 0
            llm_so_far = ""
            turn_history = self._trim_history(self._get_history(session_id))

            async def emit(ev):
                await out_q.put(("resp", turn_id, ev))

            async def synth_and_emit(sentence: str):
                nonlocal first_tts
                t0 = time.time()
                if barge_in_event.is_set():
                    return
                async for pcm in self.tts.synthesize_streaming(sentence):
                    if barge_in_event.is_set():
                        logger.info("hf run_turn: barge-in set, aborting TTS stream mid-sentence")
                        break                             # generator finally -> stop_flag, no drain
                    if len(pcm) == 0 or int(np.max(np.abs(pcm))) == 0:
                        continue                          # silence placeholder guard
                    if first_tts is None:
                        first_tts = time.time()
                        await emit({"type": "tts_start", "sampleRate": self.tts.sample_rate})
                    await emit({"type": "tts_chunk", "pcm": pcm, "text": sentence,
                                "sampleRate": self.tts.sample_rate,
                                "latency_ms": int((time.time() - t0) * 1000)})

            try:
                gen = (self.llm.generate_chat_with_tools(turn_history, txt)
                       if hasattr(self.llm, "generate_chat_with_tools")
                       else self.llm.generate_stream(txt))
                async for ev in gen:
                    et = ev.get("type")
                    if et == "llm_reset":
                        # Harness withdrew text it had streamed (XML tool call appeared /
                        # a tool step followed): drop the buffer so it never reaches TTS.
                        tts_buf = ""
                        cnt = 0
                        llm_so_far = ""
                        await emit({"type": "llm_reset"})
                        continue
                    if et in ("tool_call", "tool_result"):
                        await emit(ev)
                        continue
                    if et != "llm_token":
                        continue
                    if first_llm is None:
                        first_llm = time.time()
                    await emit(ev)
                    llm_so_far = ev.get("text_so_far", llm_so_far)
                    tok = ev.get("token", "")
                    if "<tool" in tok.lower() or ("<" in tok and ">" in tok):
                        continue                          # tool markup is not speech
                    tts_buf += tok
                    cnt += 1
                    if (SENTENCE_END.search(tok) or cnt >= 300) and tts_buf.strip():
                        s = tts_buf.strip()
                        tts_buf = ""
                        cnt = 0
                        await synth_and_emit(s)
                        if barge_in_event.is_set():
                            return
                if tts_buf.strip():
                    await synth_and_emit(tts_buf.strip())
                hist = self._get_history(session_id)
                if not hist or hist[0].get("role") != "system":
                    hist.insert(0, {"role": "system", "content": "You are helpful."})
                hist.append({"role": "user", "content": txt})
                hist.append({"role": "assistant", "content": llm_so_far})
                self.sessions[session_id] = self._trim_history(hist)
                await emit({"type": "tts_end"})
                await emit({"type": "latency", "stt_ms": stt_ms,
                            "llm_ttft_ms": int((first_llm - e2e_start) * 1000) if first_llm else 0,
                            "tts_ttfb_ms": int((first_tts - e2e_start) * 1000) if first_tts else 0,
                            "e2e_ms": int((time.time() - e2e_start) * 1000)})
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("hf_official turn failed")
                await emit({"type": "error", "error": str(e)})

        pump = asyncio.create_task(stt_pump())
        response_task = None
        current_turn_id = 0
        try:
            while True:
                item = await out_q.get()
                if item[0] == "stt":
                    ev = item[1]
                    if ev.get("type") == "_pump_done":
                        break
                    if ev.get("type") == "stt_final":
                        yield ev                                   # stt events stay untagged
                        text = (ev.get("text") or "").strip()
                        if not text:
                            continue
                        if on_new_voice_turn is not None:
                            await on_new_voice_turn()               # cancel in-flight text reply
                        if response_task is not None and not response_task.done():
                            response_task.cancel()
                        async with barge_in_lock:
                            barge_in_event.clear()                  # fresh slate for the new turn
                        current_turn_id = self.next_turn_id(session_id)
                        response_task = asyncio.create_task(
                            run_turn(text, int(ev.get("latency_ms", 0) or 0), current_turn_id))
                    elif ev.get("type") == "stt_partial":
                        yield ev
                        # Speech over a reply that is still playing == voice barge-in.
                        if (response_task is not None and not response_task.done()
                                and (ev.get("text") or "").strip()):
                            if on_new_voice_turn is not None:
                                await on_new_voice_turn()
                            response_task.cancel()
                            response_task = None
                            async with barge_in_lock:
                                barge_in_event.set()
                                barge_in_event.clear()
                            yield {"type": "barge_in", "reason": "voice", "turn_id": current_turn_id}
                    else:
                        yield ev
                else:                                    # ("resp", turn_id, ev)
                    _, ev_turn_id, ev = item
                    if ev_turn_id != current_turn_id:
                        continue                                       # stale turn
                    ev = {**ev, "turn_id": ev_turn_id}
                    yield ev
                    if ev.get("type") == "tts_end" and response_task is not None:
                        response_task = None
        finally:
            if response_task is not None and not response_task.done():
                response_task.cancel()
            if not pump.done():
                pump.cancel()
