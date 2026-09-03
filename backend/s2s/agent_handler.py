"""Our agent as the LLM stage of the HuggingFace speech-to-speech pipeline.

Replaces what ``get_llm_handler()`` returns, so s2s owns VAD, Smart Turn
endpointing, STT, TTS, transport and cancellation, while the turn itself is
still driven by this project's own harness (three tools, a bounded call ->
observe loop) and the ``ling_streaming`` text filters.

Tools run SERVER-SIDE, inside the harness, and the Realtime client-tool
protocol is deliberately left unused. That protocol expects the *client* to
execute a tool and post ``function_call_output`` back, which cannot work here:
``web_search`` talks to SearXNG on 127.0.0.1 and the weather/clock tools are
backend resources. The browser therefore sees ordinary assistant text, and the
harness's own call -> observe loop stays intact.

Contract (speech_to_speech 0.2.12):
  in   GenerateResponseRequest   -- carries runtime_config (and thus .chat)
  out  LLMResponseChunk | EndOfResponse
``LMOutputProcessor`` forwards chunks to TTS 1:1 with no sentence splitting, so
chunks emitted here must already be sentence-sized or the TTS will stutter.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import Iterator
from typing import Any

from agent.native_loop import audio_content
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk

logger = logging.getLogger(__name__)

# Flush on CJK and western sentence enders. Kept deliberately small: the harness
# has already stripped reasoning, tool XML and prompt echo by this point.
_SENTENCE_END = re.compile(r"[。！？；!?;]|(?<=[.…])\s")
_MAX_FLUSH = 120    # hard cap so a run-on clause still reaches TTS promptly
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def _squash(s: str) -> str:
    """Collapse whitespace and punctuation for "has this already been said?".

    The harness whitespace-collapses and leakage-strips its final text while the
    live deltas are raw, so a literal comparison diverges at the first newline.
    """
    return re.sub(r"[\s\u3000]+", "", s)


def _worth_speaking(chunk: str) -> bool:
    """Whether a candidate chunk carries speech, rather than punctuation noise.

    A character-count floor is wrong for Chinese, where "好。" is a complete
    sentence, while in a numbered list the "1." must NOT be flushed as its own
    utterance. So gate on content: one ideograph, or two latin letters.
    """
    return bool(_CJK.search(chunk)) or len(_LATIN.findall(chunk)) >= 2


class AgentLanguageModelHandler(BaseHandler[LLMIn, LLMOut]):
    """s2s LLM stage backed by ``LingStreaming.generate_chat_with_tools``."""

    def setup(
        self,
        api_base: str = "http://127.0.0.1:11435/v1",
        model_name: str = "bonsai-8b",
        max_new_tokens: int = 512,
        cancel_scope: Any = None,
        speculative_turns: Any = None,
        **_ignored: Any,
    ) -> None:
        from llm.ling_streaming import LingStreaming

        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.max_new_tokens = max_new_tokens
        self.api_base, self.model_name = api_base, model_name
        self.llm = LingStreaming(api_base=api_base, model_name=model_name)
        # Set by serve.py after the pipeline is built; see _give_stage_the_event_queue.
        self.text_output_queue = None

        # The harness is async; this stage is a thread. One long-lived loop in a
        # daemon thread avoids paying event-loop setup on every single turn.
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="agent-loop").start()
        logger.info("AgentLanguageModelHandler ready (%s @ %s)", model_name, api_base)

    def reconfigure(self, api_base: str, model_name: str) -> None:
        """Point the stage at a different endpoint without restarting the pipeline.

        Used by POST /v1/llm-config so the UI can switch between the local
        llama-server and a hosted provider mid-session. The adapter caches its
        reachability probe per instance, so it is rebuilt rather than mutated.
        """
        from llm.ling_streaming import LingStreaming

        self.api_base, self.model_name = api_base, model_name
        self.llm = LingStreaming(api_base=api_base, model_name=model_name)
        # Set by serve.py after the pipeline is built; see _give_stage_the_event_queue.
        self.text_output_queue = None
        logger.info("LLM stage repointed at %s (%s)", api_base, model_name)

    # -- turn gating ----------------------------------------------------
    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        """Mirror the base LLM handlers: drop output from a superseded revision."""
        if self.speculative_turns is None:
            return True
        return bool(self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision))

    def _stale(self, gen: int | None, req: LLMIn) -> bool:
        if self.cancel_scope is not None and gen is not None and self.cancel_scope.is_stale(gen):
            return True
        return not self._turn_output_allowed(req.turn_id, req.turn_revision)

    @staticmethod
    def _audio_prompt(audio, sample_rate: int):
        """The spoken turn, as content parts the loop can pass straight through."""
        import numpy as np

        a = np.asarray(audio)
        if a.dtype.kind == "f":                      # VAD hands over float32 in [-1, 1]
            a = (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16)
        return [audio_content(a.astype("<i2").tobytes(), sample_rate)]

    # -- prompt / history ------------------------------------------------
    # How many past spoken turns keep their audio. Each second of speech is ~26
    # prompt tokens, so a handful of short turns is affordable; beyond that the
    # audio is dropped and only the assistant's own replies remain, which keeps
    # the thread of the conversation without re-prefilling minutes of sound.
    _AUDIO_TURNS = 3

    @staticmethod
    def _api_audio_part(part: dict) -> dict | None:
        """Chat stores an audio part as {type, audio}; the API wants input_audio."""
        if part.get("type") != "input_audio" or not part.get("audio"):
            return None
        return {"type": "input_audio",
                "input_audio": {"data": part["audio"], "format": "wav"}}

    @classmethod
    def _spoken_history(cls, req: LLMIn) -> list[dict]:
        """Every past turn, with the user's speech kept as audio.

        The text path drops the newest user message because it is the prompt.
        Here the prompt is the audio on the request itself, so nothing in the
        Chat is the current turn and all of it is history.
        """
        chat = getattr(req.runtime_config, "chat", None)
        if chat is None:
            return []
        try:
            items = chat.to_transformers_chat()
        except Exception:
            logger.exception("could not read chat history")
            return []

        out: list[dict] = []
        for m in items:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role not in ("user", "assistant"):
                continue
            body = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if isinstance(body, list):
                parts = [q for q in (cls._api_audio_part(p) for p in body if isinstance(p, dict))
                         if q]
                text = " ".join(str(p.get("text") or "") for p in body
                                if isinstance(p, dict) and p.get("text")).strip()
                if text:
                    parts.append({"type": "text", "text": text})
                if parts:
                    out.append({"role": role, "content": parts})
            elif isinstance(body, str) and body.strip():
                out.append({"role": role, "content": body})

        # Drop audio from all but the most recent turns; keep their text.
        spoken = [i for i, m in enumerate(out) if isinstance(m["content"], list)]
        for i in spoken[:-cls._AUDIO_TURNS] if len(spoken) > cls._AUDIO_TURNS else []:
            kept = [p for p in out[i]["content"] if p.get("type") == "text"]
            out[i]["content"] = kept or "（先前的語音提問）"
        return out

    @staticmethod
    def _record_turn(req: LLMIn, wav_b64: str | None, answer: str) -> None:
        """Write the finished turn into the shared Chat.

        With --stt none nothing else does: there is no STT stage to raise a
        completed transcription, and captions are deliberately partial events
        that mutate no state. Without this the model is handed an empty history
        on every turn, so it cannot resolve "and tomorrow?" -- or even notice
        that the user just said they did NOT want the weather.
        """
        chat = getattr(req.runtime_config, "chat", None)
        if chat is None:
            return
        try:
            from speech_to_speech.LLM.chat import (
                RealtimeConversationItemAssistantMessage,
                RealtimeConversationItemUserMessage,
            )

            if wav_b64:
                chat.add_item(RealtimeConversationItemUserMessage(
                    role="user", type="message",
                    content=[{"type": "input_audio", "audio": wav_b64}]))
            if answer:
                chat.add_item(RealtimeConversationItemAssistantMessage(
                    role="assistant", type="message",
                    content=[{"type": "output_text", "text": answer}]))
            trim = getattr(chat, "trim_if_needed", None)
            if callable(trim):
                trim()
        except Exception:
            logger.exception("could not record the turn into the chat")

    @staticmethod
    def _history_and_prompt(req: LLMIn) -> tuple[list[dict], str]:
        """Read the conversation out of the shared Chat on the runtime config.

        The newest user message is the prompt; everything before it is history in
        the {role, content} shape the harness expects.
        """
        chat = getattr(req.runtime_config, "chat", None)
        if chat is None:
            return [], ""
        try:
            items = chat.to_transformers_chat()
        except Exception:
            logger.exception("could not read chat history")
            return [], ""

        # 0.2.12 returns plain {role, content} dicts even though the annotation
        # names pydantic models; accept either so a version bump cannot silently
        # empty every prompt.
        def field(m: Any, key: str) -> Any:
            return m.get(key) if isinstance(m, dict) else getattr(m, key, None)

        def text_of(m: Any) -> str:
            c = field(m, "content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # multimodal user turn; keep the text parts
                return " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict)).strip()
            return ""

        prompt = ""
        for i in range(len(items) - 1, -1, -1):
            if field(items[i], "role") == "user":
                prompt = text_of(items[i])
                items = items[:i]
                break
        history = []
        for m in items:
            role = field(m, "role")
            if role not in ("user", "assistant"):
                continue  # tool_calls / tool results stay inside the harness
            body = text_of(m)
            if body:
                history.append({"role": role, "content": body})
        return history, prompt

    # -- main loop -------------------------------------------------------
    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        gen = self.cancel_scope.generation if self.cancel_scope is not None else None
        history, prompt = self._history_and_prompt(request)
        # With --stt none there is no transcript: the turn arrives as audio on the
        # request, and the model takes it as the prompt. Our own loop carries it, so
        # unlike the framework's audio path this keeps the tools -- E4B routes one
        # from speech alone in 0.2-0.6 s.
        audio = getattr(request, "audio", None)
        if audio is not None:
            sr = getattr(request, "audio_sample_rate", 16000)
            # The caption lands in the Chat as a user text message, so overriding
            # `prompt` with audio afterwards is what keeps the model on the speech.
            # Without that, the caption would silently become the prompt and this
            # would be an ordinary STT pipeline again -- and because the caption is
            # published asynchronously, whether it arrived first would decide the
            # behaviour. Overriding makes the outcome the same either way.
            # No caption work here any more: s2s/caption.py streams it live from the
            # audio tee, so it appears while the user is still talking rather than
            # after the segment closes. This stage only has to answer.
            prompt = self._audio_prompt(audio, sr)
            # The Chat holds no user message on this path (no STT stage), so the
            # newest-message-is-the-prompt rule of the text path does not apply:
            # every stored turn is history, and the audio here is the prompt.
            history = self._spoken_history(request)

        def chunk(text: str) -> LLMResponseChunk:
            return LLMResponseChunk(
                text=text,
                language_code=request.language_code,
                runtime_config=request.runtime_config,
                response=request.response,
                turn_id=request.turn_id,
                turn_revision=request.turn_revision,
                speech_stopped_at_s=request.speech_stopped_at_s,
                cancel_generation=gen,
            )

        # A spoken turn's prompt is a list of content parts, which has no .strip().
        if not (prompt if isinstance(prompt, list) else prompt.strip()):
            logger.warning("empty prompt; nothing to answer")
            yield EndOfResponse(
                turn_id=request.turn_id, turn_revision=request.turn_revision, cancel_generation=gen
            )
            return

        pending, spoken, error = "", False, None
        answered = ""      # what actually reached the user, for the history
        try:
            for kind, payload in self._drive(history, prompt):
                if self._stale(gen, request):
                    logger.debug("turn superseded; aborting generation")
                    break
                if kind == "reset":
                    # The harness retracted what it had streamed (e.g. it turned out
                    # to be reasoning). Nothing spoken yet can be recalled, so only
                    # drop what is still buffered.
                    pending = ""
                    continue
                if kind == "error":
                    error = payload
                    break
                pending += payload
                while True:
                    cut = self._split_at(pending)
                    if cut is None:
                        break
                    head, pending = pending[:cut].strip(), pending[cut:]
                    if head:
                        spoken = True
                        answered += head
                        yield chunk(head)
            if pending.strip() and not self._stale(gen, request):
                spoken = True
                answered += pending.strip()
                yield chunk(pending.strip())
        except Exception as e:  # a dead LLM must not wedge the pipeline
            logger.exception("Qwen-Agent turn failed")
            error = str(e)

        if error and not spoken and not self._stale(gen, request):
            from agent.qwen_harness import NO_ANSWER_ZH

            yield chunk(NO_ANSWER_ZH)

        # Only a turn that actually reached the user belongs in the history; a
        # superseded (barged-in) one never happened as far as the user is
        # concerned, and recording it would answer a question they interrupted.
        if audio is not None and answered.strip() and not self._stale(gen, request):
            part = prompt[0] if isinstance(prompt, list) and prompt else {}
            self._record_turn(request, (part.get("input_audio") or {}).get("data"),
                              answered.strip())

        yield EndOfResponse(
            turn_id=request.turn_id,
            turn_revision=request.turn_revision,
            cancel_generation=gen,
            error=error,
        )

    @staticmethod
    def _split_at(buf: str) -> int | None:
        """Index to cut a speakable chunk at, or None to keep buffering."""
        for m in _SENTENCE_END.finditer(buf):
            if _worth_speaking(buf[: m.end()]):
                return m.end()
        if len(buf) >= _MAX_FLUSH:
            cut = buf.rfind(" ", 0, _MAX_FLUSH)
            return cut + 1 if cut > 0 else _MAX_FLUSH
        return None

    def _drive(self, history: list[dict], prompt: str) -> Iterator[tuple[str, str]]:
        """Pump the harness's async generator from this synchronous thread.

        Yields ("delta"|"reset"|"error", payload). Only spoken text is forwarded;
        reasoning and tool traffic are logged, never sent to TTS.

        A "reset" clears the caller's unspoken buffer. It CANNOT unspeak: anything
        already flushed to TTS has been heard. So every delta is checked against
        what has gone out -- emitting the authoritative final answer wholesale
        after a partial stream is what made answers arrive twice.
        """
        agen = self.llm.generate_chat_with_tools(history, prompt, max_new_tokens=self.max_new_tokens)
        last = ""
        emitted = ""

        def novel(text: str) -> str:
            """The part of `text` not already sent downstream."""
            nonlocal emitted
            if not text:
                return ""
            if _squash(text) in _squash(emitted):
                return ""
            common = 0
            for a, b in zip(emitted, text, strict=False):
                if a != b:
                    break
                common += 1
            tail = text[common:]
            if tail:
                emitted += tail
            return tail
        try:
            while True:
                try:
                    ev = asyncio.run_coroutine_threadsafe(agen.__anext__(), self._loop).result()
                except StopAsyncIteration:
                    return
                t = ev.get("type")
                if t == "llm_token":
                    # text_so_far is authoritative: the harness rewrites it when it
                    # retracts a span, so a delta computed from it stays correct.
                    so_far = ev.get("text_so_far") or ""
                    if not so_far.startswith(last):
                        yield "reset", ""      # drop the buffer; what was said stands
                    last = so_far
                    delta = novel(so_far)
                    if delta:
                        yield "delta", delta
                elif t == "llm_reset":
                    last = ""
                    yield "reset", ""
                elif t == "llm_done":
                    # The authoritative answer can be a filtered SUBSET of what
                    # streamed (reasoning sentences dropped), so it is not
                    # necessarily an extension of it. novel() covers both.
                    tail = novel(ev.get("text") or "")
                    if tail:
                        yield "delta", tail
                    return
                elif t == "tool_call":
                    logger.info("tool -> %s", ev.get("name"))
                elif t == "error":
                    yield "error", str(ev.get("message") or "llm error")
                    return
        finally:
            try:
                asyncio.run_coroutine_threadsafe(agen.aclose(), self._loop).result(timeout=5)
            except Exception:
                pass

    def cleanup(self) -> None:
        loop = getattr(self, "_loop", None)
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
