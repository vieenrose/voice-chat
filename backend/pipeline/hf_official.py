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
            import tempfile, soundfile as sf, os
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

    async def transcribe_stream(self, pcm_queue: asyncio.Queue, stop_event: asyncio.Event):
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
                # Yield partial every 1s
                if len(buf) > 5:
                    pcm = np.concatenate(buf)
                    pcm_f32 = pcm.astype(np.float32) / 32768.0
                    # partial not supported, just yield empty
                    yield {"type": "stt_partial", "text": "", "latency_ms": 0}
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
            except:
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
            except Exception as e:
                # Try with speaker param
                try:
                    out = self.pipe(text, forward_params={"speaker": "Chelsie"})
                    wav = out["audio"] if isinstance(out, dict) else out
                    wav = np.array(wav).squeeze()
                    return (wav * 32767).astype(np.int16)
                except:
                    raise e
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
            if not sent: continue
            async for c in self.synthesize_streaming(sent):
                yield {"type": "tts_chunk", "pcm": c, "text": sent, "sampleRate": self.sample_rate, "latency_ms": 40}
        yield {"type": "tts_end"}

    async def stream_tts(self, token_stream):
        buf=""; cnt=0
        async for ev in token_stream:
            if ev["type"]=="llm_token":
                buf+=ev["token"]; cnt+=1
                should=False
                if re.search(r'[.!?。！？\n]', ev["token"]): should=True
                elif cnt>=8 and buf and buf[-1] in " ,": should=True
                if should and buf.strip():
                    txt=buf.strip(); buf=""; cnt=0
                    async for c in self.synthesize_streaming(txt):
                        yield {"type":"tts_chunk","pcm":c,"text":txt,"sampleRate":self.sample_rate,"latency_ms":40}
            elif ev["type"]=="llm_done":
                if buf.strip():
                    async for c in self.synthesize_streaming(buf.strip()):
                        yield {"type":"tts_chunk","pcm":c,"text":buf.strip(),"sampleRate":self.sample_rate,"latency_ms":40}
                yield {"type":"tts_end"}; return
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
        if len(h) <= 1+max_turns*2: return h
        return [h[0]] + h[-(max_turns*2):]

    async def stream_chat_interleaved(self, pcm_queue, stop_event, session_id="default", barge_in_event=None, barge_in_lock=None, on_new_voice_turn=None):
        # NOTE: experimental/opt-in pipeline (HF_OFFICIAL flag, off by default) — accepts
        # barge_in_event/barge_in_lock/on_new_voice_turn for interface parity with
        # speech_to_speech.py's default pipeline (app.py's sender_loop always passes all
        # three) but does not yet implement cancellation on any of them. Without accepting
        # them here, enabling HF_OFFICIAL would TypeError on the very first WS message
        # instead of merely lacking barge-in support.
        # Reuse same logic as before but with HF components
        import re, time
        SENT_END = re.compile(r'[.!?。！？\n]')
        history = self._trim_history(self._get_history(session_id))
        e2e_start = time.time()
        async for stt_ev in self.stt.transcribe_stream(pcm_queue, stop_event):
            yield stt_ev
            if stt_ev["type"]=="stt_final":
                txt = stt_ev["text"]
                if not txt.strip(): continue
                stt_ms = stt_ev.get("latency_ms",0)
                llm_start=time.time(); tts_buf=""; cnt=0; first_llm=None; first_tts=None; llm_so_far=""
                llm_gen = self.llm.generate_chat_with_tools(history, txt)
                async for ev in llm_gen:
                    if ev["type"]=="llm_token":
                        if first_llm is None: first_llm=time.time()
                        yield ev
                        llm_so_far=ev["text_so_far"]
                        tts_buf+=ev["token"]; cnt+=1
                        should=False
                        if SENT_END.search(ev["token"]): should=True
                        elif cnt>=8 and tts_buf and tts_buf[-1] in " ,": should=True
                        if should and tts_buf.strip():
                            s=tts_buf.strip(); tts_buf=""; cnt=0
                            pcm=await self.tts.synthesize(s)
                            if first_tts is None: first_tts=time.time()
                            yield {"type":"tts_chunk","pcm":pcm,"text":s,"sampleRate":self.tts.sample_rate,"latency_ms":40}
                    elif ev["type"]=="llm_done":
                        if tts_buf.strip():
                            pcm=await self.tts.synthesize(tts_buf.strip())
                            yield {"type":"tts_chunk","pcm":pcm,"text":tts_buf.strip(),"sampleRate":self.tts.sample_rate,"latency_ms":40}
                        # Update history
                        hist=self._get_history(session_id)
                        if not hist or hist[0].get("role")!="system":
                            hist.insert(0, {"role":"system","content":"You are helpful."})
                        hist.append({"role":"user","content":txt})
                        hist.append({"role":"assistant","content":llm_so_far})
                        self.sessions[session_id]=self._trim_history(hist)
                        yield {"type":"tts_end"}
                        break
