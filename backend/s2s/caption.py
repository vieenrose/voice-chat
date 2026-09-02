"""Live captions from X-ASR, on a side path that cannot slow the pipeline.

The model hears the speech itself; this exists so the screen has something on it
while someone is still talking. Two properties are load-bearing:

**It never blocks the audio path.** ``feed()`` does one non-blocking put on an
unbounded queue and returns; all decoding happens on a daemon thread. If that
thread falls behind, captions lag -- the turn does not.

**It never enters the pipeline's state.** Captions are published as *partial*
transcription events. A completed transcription is handled by RealtimeService,
which adds the text to the Chat as a user message, i.e. it reaches the model on
the next turn; measured, that cost 0.3 s of median first-audio latency. A partial
is emitted to the client and mutates nothing (handlers/conversation.py:
on_partial_transcription touches the chat zero times).

Audio is teed from AudioHandler.append_pcm, the one funnel every inbound chunk
passes through regardless of transport, so nothing is inserted into the handler
chain and no queue hop is added.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

logger = logging.getLogger(__name__)

# Sentinels travelling the same queue as the audio, so a boundary is applied at the
# right point in the stream rather than whenever another thread happened to call.
_BEGIN, _END = object(), object()

# Decode at most this often. The recogniser is happy to be called per 32 ms chunk,
# but each call costs CPU that the TTS and the pipeline threads also want, and a
# caption that updates 30 times a second is no more readable than one updating 4.
_DECODE_EVERY_S = float(os.getenv("S2S_CAPTION_INTERVAL", "0.25"))

# Utterance boundaries are NOT decided here. begin() and end() are driven by the
# pipeline's own Silero/Smart-Turn events, so a caption covers exactly the segment
# the model is given -- the alternative, X-ASR's own endpointing or an idle timer,
# drifts from the pipeline and produces captions that describe a different span of
# audio than the answer does.


class CaptionStream:
    """Streams partial transcripts for display. Start one per process."""

    def __init__(self, text_output_queue, device: str | None = None):
        self._q: queue.Queue = queue.Queue()
        self._out = text_output_queue
        self._device = device or os.getenv("S2S_CAPTION_DEVICE", "cpu")
        self._rec = None
        self._stream = None
        self._sent = ""            # what the client has already been shown
        self._last_decode = 0.0
        self._active = False       # inside a VAD utterance: publish only then
        threading.Thread(target=self._run, daemon=True, name="xasr-caption").start()

    # -- called from the audio path; must stay O(1) --------------------
    def feed(self, pcm16: bytes, sample_rate: int) -> None:
        try:
            self._q.put_nowait((pcm16, sample_rate))
        except Exception:
            pass                   # a dropped caption chunk is not worth a raised turn

    def begin(self) -> None:
        """The pipeline's VAD reports speech. Start publishing.

        Deliberately does NOT reset the decoder. Silero confirms speech a few
        hundred ms in, and the pipeline hands the model a segment that includes that
        pre-roll -- resetting here dropped it, so the caption read 台北天气如何 for a
        「今天台北天氣如何」 the model heard in full. Audio is therefore fed
        continuously and only *publishing* is gated, which keeps the two spans equal.
        """
        self._q.put_nowait(_BEGIN)

    def end(self) -> None:
        """The pipeline's VAD reports end of speech. Flush and finish."""
        self._q.put_nowait(_END)

    # -- everything below runs on the daemon thread -------------------
    def _load(self) -> bool:
        if self._rec is not None:
            return True
        try:
            from stt.xasr_streaming import StreamingXASR

            self._rec = getattr(StreamingXASR(device=self._device), "recognizer", None)
            if self._rec is None:
                logger.warning("no sherpa backend for captions; captions disabled")
                return False
            logger.info("caption stream ready (X-ASR on %s)", self._device)
            return True
        except Exception:
            logger.exception("caption stream could not start; captions disabled")
            self._rec = False      # sentinel: do not retry every chunk
            return False

    def _publish(self, text: str) -> None:
        """Send only what is new, so the UI sees a growing line."""
        if not text or text == self._sent or self._out is None:
            return
        try:
            from speech_to_speech.pipeline.events import PartialTranscriptionEvent

            # The client treats these as cumulative and replaces, so the whole
            # current text goes each time rather than a diff.
            self._out.put(PartialTranscriptionEvent(delta=text))
            self._sent = text
        except Exception:
            logger.exception("could not publish a caption")

    def _reset(self) -> None:
        self._stream = None
        self._sent = ""

    def _run(self) -> None:
        import numpy as np

        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            # Both boundaries travel this one queue with the audio, so they are
            # applied at the right point in the stream rather than whenever another
            # thread happened to call.
            if item is _BEGIN:
                self._active = True
                self._sent = ""
                continue
            if item is _END:
                self._flush()
                self._active = False
                self._reset()
                continue
            if self._rec is False or (self._rec is None and not self._load()):
                continue
            pcm, sr = item
            try:
                a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                if self._stream is None:
                    self._stream = self._rec.create_stream()
                self._stream.accept_waveform(sr, a)
                now = time.monotonic()
                if now - self._last_decode < _DECODE_EVERY_S:
                    continue
                self._last_decode = now
                while self._rec.is_ready(self._stream):
                    self._rec.decode_stream(self._stream)
                # Between utterances the decoder still receives audio (to keep the
                # pre-roll) but nothing is shown: silence and room noise would
                # otherwise publish junk before the user has said anything.
                if self._active:
                    self._publish((self._rec.get_result(self._stream) or "").strip())
            except Exception:
                logger.exception("caption decode failed; resetting the stream")
                self._reset()

    def _flush(self) -> None:
        """Decode whatever is buffered when the talking stops."""
        if self._stream is None or not self._active:
            return
        try:
            while self._rec.is_ready(self._stream):
                self._rec.decode_stream(self._stream)
            self._publish((self._rec.get_result(self._stream) or "").strip())
        except Exception:
            logger.exception("caption flush failed")
