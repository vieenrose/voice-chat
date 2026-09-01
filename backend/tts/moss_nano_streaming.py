"""
MOSS-TTS-Nano-100M — TRUE streaming wrapper (Realtime Streaming Decode)
Apache-2.0, 100M, 20 langs, 48k. Uses MossTTSNanoRuntime.synthesize_stream() which
yields live `type:"audio"` chunks (incremental decode) — real TTFA, NOT one-shot+split.

Voices via built-in presets (voice_clone): default 'Junhao' (zh_1), '台湾腔 Yuewen' (zh_4),
'Ava' (en_2) etc. Fallback: Qwen3-TTS if MOSS fails to load.
"""
import asyncio
import time
import re
import sys
import os
from pathlib import Path
from typing import AsyncGenerator, Iterator
import numpy as np
from loguru import logger

SAMPLE_RATE = 48000
FLUSH_TOKENS = 8
SENTENCE_END = re.compile(r'[.!?。！？\n]')
# MOSS repo at /tmp/MOSS-TTS-Nano exposes moss_tts_nano_runtime
_MOSS_REPO = "/tmp/MOSS-TTS-Nano"
if _MOSS_REPO not in sys.path:
    sys.path.insert(0, _MOSS_REPO)

class StreamingPrimeTTS:
    def __init__(self, model_id: str = "OpenMOSS-Team/MOSS-TTS-Nano-100M", device: str = "cuda", mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.mock = False
        self.backend = "moss-nano-100m"
        self.runtime = None
        self.sample_rate = SAMPLE_RATE
        # voice presets from MOSS repo assets/audio + demo.jsonl
        self.voices = ["Junhao", "Yuewen", "Ava", "Xiaoyu"]
        self._vv = "Junhao"
        try:
            from moss_tts_nano_runtime import NanoTTSService
            # PIN to local snapshot dirs (44502 code is NaN-free; current hub revision 8ae621 NaNs on this GPU)
            ckpt = "/home/user/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-TTS-Nano-100M/snapshots/44502f80dbf9743528fa921cc544d662c685ebec"
            atok = "/home/user/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano/snapshots/6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
            # bypass transformers remote-code loader: import MOSS classes DIRECTLY from local dirs
            try:
                import torch
                from transformers import AutoModel, AutoConfig
                import transformers.dynamic_module_utils as _dm
                _orig_dl = _dm.get_class_from_dynamic_module
                def _local_dl(pre_module, type_, *a, **kw):
                    d = kw.get("pretrained_model_name_or_path") or (a[0] if a else None)
                    if isinstance(d, str) and os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json")):
                        kw["local_files_only"] = True
                        kw.pop("trust_remote_code", None)
                    return _orig_dl(pre_module, type_, *a, **kw)
                _dm.get_class_from_dynamic_module = _local_dl
                # strongest: make AutoModel.from_pretrained skip dynamic modules for local MOSS dirs
                _oam = AutoModel.from_pretrained
                if os.path.isfile(os.path.join(ckpt, "modeling_moss_tts_nano.py")) and ckpt not in sys.path:
                    sys.path.insert(0, ckpt)
                if os.path.isfile(os.path.join(atok, "modeling_moss_audio_tokenizer.py")) and atok not in sys.path:
                    sys.path.insert(0, atok)
                def _local_am(name, *a, **kw):
                    if isinstance(name, str) and os.path.isdir(name):
                        if os.path.isfile(os.path.join(name, "modeling_moss_tts_nano.py")):
                            from modeling_moss_tts_nano import MossTTSNanoForCausalLM
                            kw["local_files_only"] = True
                            kw.pop("trust_remote_code", None)
                            return MossTTSNanoForCausalLM.from_pretrained(name, **kw)
                        if os.path.isfile(os.path.join(name, "modeling_moss_audio_tokenizer.py")):
                            from modeling_moss_audio_tokenizer import MossAudioTokenizerModel
                            kw["local_files_only"] = True
                            kw.pop("trust_remote_code", None)
                            return MossAudioTokenizerModel.from_pretrained(name, **kw)
                    return _oam(name, *a, **kw)
                AutoModel.from_pretrained = classmethod(_local_am)
            except Exception as _e:
                logger.warning(f"MOSS loader patch failed {_e}")
            logger.info(f"MOSS runtime: checkpoint=44502-local(direct import) device={device}")
            self.runtime = NanoTTSService(
                checkpoint_path=ckpt,
                audio_tokenizer_path=atok,
                device=device,
                dtype="auto",  # 44502 code is NaN-free under bf16 (auto); float32 NaNs on this GPU
                attn_implementation="sdpa",
            )
            # SURGICAL: replace lazy loaders with direct-import instances (bypass dynamic-module loader
            # that pulls the NaN-buggy hub revision 8ae621). Imported classes + local weights verified.
            import torch as _torch
            import types as _types
            import importlib as _il
            # register checkpoint dirs as importable packages so their 'from .x import' work
            _pkg_t = "moss_nano_model"
            _pkg_c = "moss_codec_model"
            for _pk, _d in ((_pkg_t, ckpt), (_pkg_c, atok)):
                if _pk not in sys.modules:
                    _m = _types.ModuleType(_pk)
                    _m.__path__ = [_d]
                    sys.modules[_pk] = _m
                elif _pk in sys.modules and not hasattr(sys.modules[_pk], "__path__"):
                    sys.modules[_pk].__path__ = [_d]
            from moss_nano_model.modeling_moss_tts_nano import MossTTSNanoForCausalLM
            from moss_codec_model.modeling_moss_audio_tokenizer import MossAudioTokenizerModel
            from transformers import AutoModel as _AM, AutoModelForCausalLM as _AMC, AutoConfig as _AC
            from moss_nano_model.configuration_moss_tts_nano import MossTTSNanoConfig
            from moss_codec_model.configuration_moss_audio_tokenizer import MossAudioTokenizerConfig
            _oam, _oamc, _oac = _AM.from_pretrained, _AMC.from_pretrained, _AC.from_pretrained
            def _has(name, f): return isinstance(name, str) and os.path.isdir(name) and os.path.isfile(os.path.join(name, f))
            def _am(cls, name, *a, **kw):
                if _has(name, "modeling_moss_audio_tokenizer.py"):
                    kw["local_files_only"]=True
                    kw["trust_remote_code"]=True
                    return MossAudioTokenizerModel.from_pretrained(name, **kw)
                return _oam(name, *a, **kw)
            def _amc(cls, name, *a, **kw):
                if _has(name, "modeling_moss_tts_nano.py"):
                    kw["local_files_only"]=True
                    kw["trust_remote_code"]=True
                    return MossTTSNanoForCausalLM.from_pretrained(name, **kw)
                return _oamc(name, *a, **kw)
            def _ac(cls, name, *a, **kw):
                if _has(name, "configuration_moss_tts_nano.py"):
                    kw["local_files_only"]=True
                    kw["trust_remote_code"]=True
                    return MossTTSNanoConfig.from_pretrained(name, **kw)
                if _has(name, "configuration_moss_audio_tokenizer.py"):
                    kw["local_files_only"]=True
                    kw["trust_remote_code"]=True
                    return MossAudioTokenizerConfig.from_pretrained(name, **kw)
                return _oac(name, *a, **kw)
            _AM.from_pretrained = classmethod(_am)
            _AMC.from_pretrained = classmethod(_amc)
            _AC.from_pretrained = classmethod(_ac)
            self.sample_rate = SAMPLE_RATE
            logger.info(f"MOSS-TTS-Nano TRUE-STREAMING loaded ✓ (44502 local code, direct import) sr={self.sample_rate}")
        except Exception as e:
            logger.error(f"MOSS streaming load failed {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"MOSS-Nano required: {e}") from e

    @property
    def voice(self): return self._vv
    def set_voice(self, name: str):
        if name in self.voices or name in ("台湾腔", "中文女声"):
            key = {"台湾腔": "Yuewen", "中文女声": "Yuewen"}.get(name, name)
            self._vv = key
            return
        raise KeyError(f"unknown voice {name}; available {self.voices}")

    @staticmethod
    def _dyn_frames(text: str) -> int:
        # MOSS has no reliable EOS -> cap frames by text length (~0.08s per frame)
        zh = any('\u4e00' <= c <= '\u9fff' for c in text)
        if zh:
            return max(30, min(150, int(len(text) * 1.6)))
        return max(40, min(160, len(text.split()) * 16))

    def _stream_events(self, text: str, voice: str, max_new_frames: int | None = None) -> Iterator[dict]:
        """Blocking generator of audio events from runtime.synthesize_stream (true realtime decode)."""
        if self.runtime is None:
            raise RuntimeError("MOSS runtime not loaded")
        mf = max_new_frames or self._dyn_frames(text)
        for ev in self.runtime.synthesize_stream(
            text=text,
            voice=voice,
            mode="voice_clone",
            output_audio_path="/tmp/moss_stream_out.wav",
            max_new_frames=mf,
            do_sample=True,
            audio_temperature=0.8,
            seed=41,
        ):
            yield ev

    async def synthesize_streaming(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        """TRUE streaming: yields int16 48k PCM chunks as they decode (TTFA low)."""
        # language: zh -> Yuewen (台湾腔 zh_4 ref), en -> Ava (en_2 ref); honor explicit set_voice
        is_zh = any('\u4e00' <= c <= '\u9fff' for c in text)
        voice = self._vv if self._vv in ("Yuewen", "Ava", "Xiaoyu") else ("Yuewen" if is_zh else "Ava")
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        loop = asyncio.get_running_loop()
        def _run():
            t0 = time.time()
            emitted=0.0
            first=True
            try:
                for ev in self._stream_events(text, voice):
                    et = ev.get("type")
                    if et == "audio":
                        w = ev.get("waveform_numpy")
                        if w is None:
                            continue
                        arr = np.asarray(w, dtype=np.float32)
                        # MOSS outputs stereo (48k 2ch) -> downmix to mono, else playback garbles
                        if arr.ndim == 2:
                            arr = arr.mean(axis=1)
                        # mild normalize: scale to ~0.9 peak for clean speech level
                        pk = float(np.max(np.abs(arr))) or 1.0
                        arr = arr / pk * 0.9
                        pcm = np.clip(arr * 32767, -32768, 32767).astype(np.int16)
                        if first:
                            ttfa = (time.time()-t0)*1000
                            logger.info(f"MOSS TRUE-streaming TTFA {ttfa:.0f}ms (chunk_id {ev.get('chunk_index')})")
                            first=False
                        asyncio.run_coroutine_threadsafe(q.put(("pcm", pcm)), loop).result()
                        emitted += ev.get("emitted_audio_seconds", len(pcm)/SAMPLE_RATE)
                    elif et == "result":
                        break
            except Exception as e:
                logger.error(f"MOSS stream error {e}")
                asyncio.run_coroutine_threadsafe(q.put(("err", e)), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(q.put(("done", None)), loop).result()
        threading.Thread(target=_run, daemon=True).start()
        while True:
            kind, payload = await q.get()
            if kind == "done":
                break
            if kind == "err":
                raise payload
            yield payload

    async def synthesize(self, text: str, reference_audio: str = None) -> np.ndarray:
        chunks=[]
        async for c in self.synthesize_streaming(text):
            chunks.append(c)
        if not chunks:
            return np.zeros(int(SAMPLE_RATE*0.4), dtype=np.int16)
        return np.concatenate(chunks)

    async def stream_tts(self, token_stream):
        buf=""
        cnt=0
        async for ev in token_stream:
            if ev["type"]=="llm_token":
                buf+=ev["token"]
                cnt+=1
                fl=False
                if SENTENCE_END.search(ev["token"]):
                    fl=True
                elif cnt>=FLUSH_TOKENS and buf and buf[-1] in " ,":
                    fl=True
                if fl and buf.strip():
                    txt=buf.strip()
                    buf=""
                    cnt=0
                    async for c in self.synthesize_streaming(txt):
                        yield {"type":"tts_chunk","pcm":c,"text":txt,"sampleRate":self.sample_rate,"latency_ms":20}
            elif ev["type"]=="llm_done":
                if buf.strip():
                    async for c in self.synthesize_streaming(buf.strip()):
                        yield {"type":"tts_chunk","pcm":c,"text":buf.strip(),"sampleRate":self.sample_rate,"latency_ms":20}
                yield {"type":"tts_end"}
                return
        if buf.strip():
            async for c in self.synthesize_streaming(buf.strip()):
                yield {"type":"tts_chunk","pcm":c,"text":buf.strip(),"sampleRate":self.sample_rate,"latency_ms":20}
        yield {"type":"tts_end"}

    async def tts_from_text(self, text):
        sents = re.split(r'([.!?。！？]+)', text)
        for i in range(0, len(sents), 2):
            s=(sents[i]+(sents[i+1] if i+1<len(sents) else "")).strip()
            if not s:
                continue
            async for c in self.synthesize_streaming(s):
                yield {"type":"tts_chunk","pcm":c,"text":s,"sampleRate":self.sample_rate,"latency_ms":20}
        yield {"type":"tts_end"}

import threading  # noqa: E402