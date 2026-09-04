"""Live captions from a second Gemma 4 E2B, on a side path that cannot slow the pipeline.

s2s/caption.py's X-ASR engine reads the same audio the model hears, but through a
much smaller, unrelated model -- so the on-screen caption can quietly diverge from
what Gemma 4 E4B actually understood: a lightweight streaming transducer misparses
a phrase the E4B answer shows it heard correctly, and the caption looks like the
bot misheard when it didn't. This engine asks a second, small Gemma 4 (E2B, its own
QAT + MTP GGUF, native audio input -- see s2s/deploy/llm-e2b-caption.sh) to
transcribe the same audio, so the caption reflects what a member of the same model
family heard, not an unrelated ASR's guess.

Runs against its own llama-server, a separate process and port from the one
driving the conversation (E4B), so a slow caption call can only make the caption
late -- it cannot block the turn.

Trades X-ASR's true incremental decoding for periodic full-buffer
re-transcription: a chat-completions call has no notion of "extend this partial",
so each tick re-sends the utterance-so-far from the start. The interval
(_DECODE_EVERY_S) is therefore coarser than X-ASR's 0.25 s, to keep the number of
calls -- and their cost -- bounded.

A short rolling pre-roll is kept between utterances and seeded into the buffer at
begin(), for the same reason X-ASR's begin() does not reset its decoder: Silero
confirms speech a few hundred ms after it actually starts, and the pipeline hands
the model a segment that includes that lead-in. Without the pre-roll the caption
and the model would describe different spans of the same audio.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

import httpx

logger = logging.getLogger(__name__)

_BEGIN, _END = object(), object()

# An LLM call costs far more than a transducer step, so ticks are coarser than
# X-ASR's 0.25 s -- frequent enough to look live, rare enough not to queue up.
_DECODE_EVERY_S = float(os.getenv("S2S_CAPTION_GEMMA_INTERVAL", "0.7"))
_API_BASE = os.getenv("S2S_CAPTION_LLM_API_BASE", "http://127.0.0.1:11436/v1")
_MODEL = os.getenv("S2S_CAPTION_LLM_MODEL", "gemma-4-e2b-caption")
_TIMEOUT = float(os.getenv("S2S_CAPTION_LLM_TIMEOUT", "8"))
_PRE_ROLL_S = float(os.getenv("S2S_CAPTION_PRE_ROLL", "0.6"))
# Safety valve if end() is ever missed: stop growing the buffer past this many
# seconds and keep only the most recent audio, so one stuck utterance cannot make
# every following call slower and slower.
_MAX_BUFFER_S = float(os.getenv("S2S_CAPTION_MAX_BUFFER", "20"))

_PROMPT = (
    "Transcribe exactly what is said in this audio clip, verbatim, in the "
    "language it is spoken in. If nothing is said, the transcript is empty."
)

# Left free to narrate, E2B answers a bare "transcribe this" prompt with a
# restated instruction and the transcript buried mid-paragraph in quotes --
# e.g. 'The audio clip is: "帮我转接陈一军"\nI need to output only the transcript.'
# -- burning the token budget on prose no caption needs. A schema-constrained
# JSON reply removes the narration outright rather than trying to parse around
# it: the grammar the server derives from this schema cannot produce prose,
# only {"transcript": "..."}.
_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "transcript",
        "schema": {
            "type": "object",
            "properties": {"transcript": {"type": "string"}},
            "required": ["transcript"],
            "additionalProperties": False,
        },
    },
}


class GemmaCaptionStream:
    """Streams periodic transcripts for display, from a second Gemma 4 E2B.

    Same feed()/begin()/end() contract as s2s.caption.CaptionStream, so the two
    are interchangeable at the call site in serve.py. Start one per process.
    """

    def __init__(self, text_output_queue, api_base: str | None = None):
        self._q: queue.Queue = queue.Queue()
        self._out = text_output_queue
        self._api_base = (api_base or _API_BASE).rstrip("/")
        self._client = httpx.Client(timeout=_TIMEOUT)
        self._sent = ""             # what the client has already been shown
        self._buf = bytearray()     # this utterance's audio, from the pre-roll on
        self._pre_roll = bytearray()
        self._rate = 16000
        self._last_decode = 0.0
        self._active = False        # inside a VAD utterance: publish only then
        threading.Thread(target=self._run, daemon=True, name="gemma-e2b-caption").start()

    # -- called from the audio path; must stay O(1) --------------------
    def feed(self, pcm16: bytes, sample_rate: int) -> None:
        try:
            self._q.put_nowait((pcm16, sample_rate))
        except Exception:
            pass                    # a dropped caption chunk is not worth a raised turn

    def begin(self) -> None:
        self._q.put_nowait(_BEGIN)

    def end(self) -> None:
        self._q.put_nowait(_END)

    # -- everything below runs on the daemon thread -------------------
    def _publish(self, text: str) -> None:
        if not text or text == self._sent or self._out is None:
            return
        try:
            from speech_to_speech.pipeline.events import PartialTranscriptionEvent

            self._out.put(PartialTranscriptionEvent(delta=text))
            self._sent = text
        except Exception:
            logger.exception("could not publish a caption")

    def _transcribe(self) -> str:
        from agent.native_loop import audio_content

        body = {
            "model": _MODEL,
            "messages": [{"role": "user",
                          "content": [{"type": "text", "text": _PROMPT},
                                      audio_content(bytes(self._buf), self._rate)]}],
            "temperature": 0.0,
            "max_tokens": 64,
            "stream": False,
            "response_format": _SCHEMA,
        }
        r = self._client.post(f"{self._api_base}/chat/completions", json=body)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"] or ""
        try:
            import json
            return (json.loads(content).get("transcript") or "").strip()
        except (json.JSONDecodeError, AttributeError):
            logger.warning("caption reply was not the expected JSON: %r", content)
            return content.strip()

    def _cap_buffer(self) -> None:
        limit = int(_MAX_BUFFER_S * self._rate) * 2   # 16-bit samples
        if len(self._buf) > limit:
            del self._buf[:-limit]

    def _run(self) -> None:
        pre_roll_bytes = int(_PRE_ROLL_S * self._rate) * 2
        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is _BEGIN:
                self._active = True
                self._sent = ""
                self._buf = bytearray(self._pre_roll)
                self._last_decode = 0.0
                continue
            if item is _END:
                self._decode()
                self._active = False
                self._pre_roll.clear()
                continue
            pcm, sr = item
            self._rate = sr
            pre_roll_bytes = int(_PRE_ROLL_S * sr) * 2
            if self._active:
                self._buf.extend(pcm)
                self._cap_buffer()
            else:
                # Between utterances: keep only a short tail, for the next begin().
                self._pre_roll.extend(pcm)
                if len(self._pre_roll) > pre_roll_bytes:
                    del self._pre_roll[:-pre_roll_bytes]
                continue
            now = time.monotonic()
            if now - self._last_decode < _DECODE_EVERY_S:
                continue
            self._last_decode = now
            self._decode()

    def _decode(self) -> None:
        if not self._buf:
            return
        try:
            self._publish(self._transcribe())
        except Exception:
            logger.exception("caption transcription failed; will retry next tick")
