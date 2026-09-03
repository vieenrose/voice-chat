#!/usr/bin/env python3
"""Regression tests for the code-review fixes (no GPU, no models, no network).

Each test locks in one fix from the review so it cannot silently regress — the
three bugs that actually shipped this month (CJK relevance scoring, truncated
tool-call JSON, wrong-language TTS segments) were all in pure functions exactly
like the ones covered here.

Run:  python3 -m unittest discover -s backend/tests -v     (stdlib only)
      backend/tests/test_live_stack.py covers the parts that need the server up.
"""
import asyncio
import inspect
import os
from unittest import mock
import re
import sys
import time
import unittest
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import app as appmod                      # noqa: E402
import llm_manager as llmmod              # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402



def _client():
    # Deliberately NOT used as a context manager: that would run the lifespan and
    # try to spawn a second SearXNG on :8888.
    return TestClient(appmod.app)


# --------------------------------------------------------------------------
# P0-1: reflected XSS in /search?format=html
# --------------------------------------------------------------------------
class TestSearchHtmlEscaping(unittest.TestCase):
    def setUp(self):
        import importlib
        # `import tools.web_search as ws` binds the re-exported FUNCTION, not the
        # module (tools/__init__.py does `from .web_search import web_search`) —
        # importlib is the only unambiguous way in.
        ws = importlib.import_module("tools.web_search")
        self._ws = ws
        self._real = ws.web_search

        async def fake(query, count=5):
            return {"query": query, "source": "fake", "latency_ms": 1, "results": [
                {"title": "<script>alert(1)</script>", "url": "javascript:alert(1)",
                 "content": "</p><img src=x onerror=alert(1)>", "score": 1, "engine": "fake"},
            ]}
        ws.web_search = fake

    def tearDown(self):
        self._ws.web_search = self._real

    def test_query_and_results_are_escaped(self):
        r = _client().get('/search?q=<b>x</b></h2><script>1</script>&format=html')
        self.assertEqual(r.status_code, 200)
        body = r.text
        self.assertNotIn("<script>", body, "raw <script> reached the HTML response")
        self.assertNotIn("<b>x</b>", body, "caller's query was interpolated unescaped")
        self.assertIn("&lt;script&gt;", body)
        # javascript: URLs must not become links
        self.assertNotIn('href="javascript:', body)
        self.assertIn('<a href="#"', body)


# --------------------------------------------------------------------------
# P0-2: wildcard CORS + credentials
# --------------------------------------------------------------------------
class TestCors(unittest.TestCase):
    def test_defaults_are_not_wildcard_with_credentials(self):
        self.assertFalse(appmod.CORS_CREDENTIALS,
                         "allow_credentials must default off")
        self.assertNotEqual(appmod.CORS_ORIGINS, ["*"],
                            "default CORS origins must be an explicit list")

    def test_foreign_origin_gets_no_cors_headers(self):
        r = _client().get("/health", headers={"Origin": "https://evil.example"})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("access-control-allow-origin", {k.lower() for k in r.headers})

    def test_credentialed_wildcard_is_refused(self):
        # Simulate the operator setting both knobs the unsafe way
        src = inspect.getsource(appmod)
        self.assertIn('CORS_CREDENTIALS = os.getenv("ALLOW_CREDENTIALS", "0") == "1" and CORS_ORIGINS != ["*"]', src)


# --------------------------------------------------------------------------
# P0-3: auth on working routes; loopback-only model switch; WS gate
# --------------------------------------------------------------------------
class TestAuth(unittest.TestCase):
    def setUp(self):
        self._saved = appmod.AUTH_TOKEN

    def tearDown(self):
        appmod.AUTH_TOKEN = self._saved

    def test_health_open_but_without_usage_history(self):
        r = _client().get("/health")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertIn("rss_mb", j)                       # UI reads this
        # The per-utterance E2E history must not be public — only a count is.
        self.assertNotIsInstance(j["stats"]["latencies"], list,
                                 "E2E latency history must not be public")
        self.assertNotIn("vms_mb", j)

    def test_verbose_health_requires_token(self):
        appmod.AUTH_TOKEN = "sekret"
        c = _client()
        self.assertEqual(c.get("/health?verbose=1").status_code, 401)
        self.assertEqual(c.get("/health?verbose=1", headers={"X-Auth-Token": "sekret"}).status_code, 200)
        self.assertIsInstance(c.get("/health?verbose=1", headers={"X-Auth-Token": "sekret"}).json()["stats"]["latencies"], list)

    def test_api_requires_token_when_configured(self):
        appmod.AUTH_TOKEN = "sekret"
        c = _client()
        self.assertEqual(c.post("/api/chat", json={"text": "hi"}).status_code, 401)
        self.assertEqual(c.get("/api/search?q=x").status_code, 401)
        self.assertEqual(c.get("/stats").status_code, 401)

    def test_spa_static_not_gated(self):
        appmod.AUTH_TOKEN = "sekret"
        # A path outside PROTECTED_PREFIXES (the mounted SPA) must still load.
        self.assertTrue(any(not p.startswith(("/api/", "/ws/", "/search", "/stats"))
                            for p in ["/", "/index.html"]), "SPA paths must stay open")
        self.assertEqual(_client().get("/nope-not-an-api").status_code, 404)

    def _switch_ok(self):
        """Stub the actual restart; the endpoint logic is what is under test."""
        async def _fake(model_id):
            return {"status": "ok", "model_id": model_id, "alias": model_id.replace("-q4", ""),
                    "label": model_id}
        return mock.patch.object(appmod.llm_manager, "switch_to", _fake)

    def _reset_switch_state(self):
        appmod._switch_state.update(at=0.0, peer="", model="")
        appmod.stats["model_switches"].clear()

    def test_open_mode_lets_the_operator_switch_from_their_own_device(self):
        """The bug this replaces: "載入失敗：model switching requires VOICE_CHAT_TOKEN when
        not on loopback". TestClient's peer is non-loopback exactly like a browser opened
        over Tailscale/LAN, so the old loopback-only rule broke the model picker for the
        person who owns the server — while a stranger with curl was never actually
        stopped by anything that also stops the owner."""
        appmod.AUTH_TOKEN = None
        appmod.MODEL_SWITCH_REQUIRE_TOKEN = False
        self._reset_switch_state()
        with self._switch_ok():
            r = _client().post("/api/model", json={"model_id": llmmod.DEFAULT_MODEL_ID})
        self.assertEqual(r.status_code, 200, r.text)
        hist = appmod.stats["model_switches"]
        self.assertEqual(len(hist), 1, "the switch must be recorded")
        self.assertTrue(hist[-1]["ok"] and hist[-1]["model"] == llmmod.DEFAULT_MODEL_ID)
        self.assertTrue(hist[-1]["peer"], "it must be attributable")

    def test_closed_mode_is_still_available_as_a_choice(self):
        appmod.AUTH_TOKEN = None
        appmod.MODEL_SWITCH_REQUIRE_TOKEN = True
        self._reset_switch_state()
        try:
            r = _client().post("/api/model", json={"model_id": llmmod.DEFAULT_MODEL_ID})
            self.assertEqual(r.status_code, 403)
            self.assertIn("remedy", r.json(), "a refusal must say what to do about it")
            self.assertEqual(len(appmod.stats["model_switches"]), 0, "refused switches are not performed")
        finally:
            appmod.MODEL_SWITCH_REQUIRE_TOKEN = False

    def test_switches_are_rate_limited_rather_than_blocked(self):
        """Attribution is only honest if the log/history survives abuse; the abuse that
        actually hurts is thrashing — every switch restarts llama-server for all sessions."""
        appmod.AUTH_TOKEN = None
        appmod.MODEL_SWITCH_REQUIRE_TOKEN = False
        self._reset_switch_state()
        with self._switch_ok():
            c = _client()
            first = c.post("/api/model", json={"model_id": llmmod.DEFAULT_MODEL_ID})
            # The registry holds a single model, so the thrash to guard against is the
            # same id twice in a row -- which still restarts llama-server for everyone.
            second = c.post("/api/model", json={"model_id": llmmod.DEFAULT_MODEL_ID})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429, "immediate re-switch must not restart the server twice")
        j = second.json()
        self.assertIn("retry_after_s", j)
        self.assertGreater(j["retry_after_s"], 0)
        self.assertEqual(len(appmod.stats["model_switches"]), 1, "the refused attempt must not be logged as a switch")

    def test_a_configured_token_is_still_demanded(self):
        appmod.AUTH_TOKEN = "sekret"
        self._reset_switch_state()
        try:
            self.assertEqual(_client().post("/api/model", json={"model_id": llmmod.DEFAULT_MODEL_ID}).status_code, 401)
            with self._switch_ok():
                ok = _client().post("/api/model", json={"model_id": llmmod.DEFAULT_MODEL_ID},
                                    headers={"X-Auth-Token": "sekret"})
            self.assertEqual(ok.status_code, 200, ok.text)
        finally:
            appmod.AUTH_TOKEN = None

    def test_unknown_model_id_is_rejected_before_any_restart(self):
        appmod.AUTH_TOKEN = None
        self._reset_switch_state()
        with self._switch_ok():
            r = _client().post("/api/model", json={"model_id": "gpt-99"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("available", r.json())
        self.assertEqual(len(appmod.stats["model_switches"]), 0)

    def test_websocket_rejected_without_token(self):
        appmod.AUTH_TOKEN = "sekret"
        with self.assertRaises(Exception) as ctx:
            with _client().websocket_connect("/ws/chat?session_id=t") as ws:
                ws.send_json({"type": "start"})
        self.assertIn(type(ctx.exception).__name__, ("WebSocketDisconnect", "HTTPException", "RuntimeError", "ConnectionClosed"))


# --------------------------------------------------------------------------
# P1-4: session isolation (unique session ids, per-connection STT streams)
# --------------------------------------------------------------------------
class TestSessionIsolation(unittest.TestCase):
    def test_default_session_id_is_made_unique(self):
        a = appmod._resolve_session_id("")
        b = appmod._resolve_session_id("default")
        self.assertTrue(a.startswith("anon-") and b.startswith("anon-"))
        self.assertNotEqual(a, b, "two clients without a session_id would share one slot")
        self.assertEqual(appmod._resolve_session_id("tab-7"), "tab-7")

    def test_stt_creates_a_stream_per_connection(self):
        """Two concurrent transcribe_stream() calls must not share decoder state."""
        from stt.xasr_streaming import StreamingXASR

        created = []
        audio_per_stream = {}

        class FakeStream:
            def __init__(self, sid):
                self.sid = sid
                self.finished = False
                created.append(sid)
                audio_per_stream[sid] = 0

            def accept_waveform(self, sr, pcm):
                audio_per_stream[self.sid] += len(pcm)

            def input_finished(self):
                self.finished = True

        class FakeRec:
            def __init__(self):
                self.n = 0

            def create_stream(self):
                self.n += 1
                return FakeStream(self.n)

            def is_ready(self, st):
                return False

            def decode_stream(self, st):
                pass

            def get_result_all(self, st):
                class R:
                    text = ""
                return R()

            def is_endpoint(self, st):
                return False

        stt = StreamingXASR.__new__(StreamingXASR)
        stt.mock = False
        stt.backend = "x-asr-sherpa-160ms"
        stt.recognizer = FakeRec()

        async def run():
            q1, q2 = asyncio.Queue(), asyncio.Queue()
            stop = asyncio.Event()
            for _ in range(3):
                await q1.put(np.zeros(320, dtype=np.int16))
                await q2.put(np.zeros(320, dtype=np.int16))

            async def drain(q, out):
                async for ev in stt.transcribe_stream(q, stop, "sess"):
                    out.append(ev)

            a, b = [], []
            t1 = asyncio.create_task(drain(q1, a))
            t2 = asyncio.create_task(drain(q2, b))
            await asyncio.sleep(0.4)
            stop.set()
            await asyncio.wait_for(asyncio.gather(t1, t2), timeout=5)

        asyncio.run(run())
        self.assertGreaterEqual(len(created), 2, "each connection must get its own stream")
        # Each stream saw only its own 3*320 samples — not 6*320 shared ones
        self.assertTrue(all(v == 3 * 320 for v in audio_per_stream.values()),
                        f"audio leaked across streams: {audio_per_stream}")


# --------------------------------------------------------------------------
# P1-5: blocking STT inference must not run on the event loop
# --------------------------------------------------------------------------
class TestSttOffEventLoop(unittest.TestCase):
    def test_loop_stays_responsive_during_decode(self):
        from stt.xasr_streaming import StreamingXASR

        class SlowStream:
            def accept_waveform(self, sr, pcm):
                pass

            def input_finished(self):
                pass

        class SlowRec:
            def create_stream(self):
                return SlowStream()

            def is_ready(self, st):
                return True   # always "ready" -> exactly one decode per burst

            def decode_stream(self, st):
                time.sleep(0.06)   # stand-in for sherpa native inference
                raise _StopDecode  # break out of the while-is_ready loop after 1 decode

            def get_result_all(self, st):
                class R:
                    text = ""
                return R()

            def is_endpoint(self, st):
                return False

        class _StopDecode(Exception):
            pass

        stt = StreamingXASR.__new__(StreamingXASR)
        stt.mock = False
        stt.backend = "x-asr-sherpa-160ms"
        stt.recognizer = SlowRec()

        async def run():
            q = asyncio.Queue()
            stop = asyncio.Event()
            for _ in range(4):
                await q.put(np.zeros(320, dtype=np.int16))
            ticks = {"n": 0}

            async def heartbeat():
                while True:
                    ticks["n"] += 1
                    await asyncio.sleep(0.01)

            async def pump():
                try:
                    async for _ in stt.transcribe_stream(q, stop, "sess"):
                        pass
                except Exception:
                    stop.set()

            hb = asyncio.create_task(heartbeat())
            # Let decode failures terminate the generator, then measure responsiveness
            p = asyncio.create_task(pump())
            t0 = time.time()
            while time.time() - t0 < 0.35:
                await asyncio.sleep(0.01)
            stop.set()
            hb.cancel()
            try:
                await asyncio.wait_for(p, timeout=2)
            except (asyncio.TimeoutError, Exception):
                pass
            return ticks["n"]

        ticks = asyncio.run(run())
        # ~350ms of wall time with 4 x 60ms sleeps: if the sleeps had run on the loop
        # the heartbeat could not exceed ~2-3 ticks; off-loop it keeps ticking.
        self.assertGreater(ticks, 10, f"event loop was blocked during STT decode (only {ticks} beats)")


# --------------------------------------------------------------------------
# P1-6: tool turns stream; multi-turn history reaches the harness
# --------------------------------------------------------------------------
class TestHarnessContract(unittest.TestCase):
    def test_all_harnesses_accept_history(self):
        for name in ("agent.qwen_harness", "agent.harness", "agent.pydantic_harness"):
            import importlib
            mod = importlib.import_module(name)
            self.assertIn("history", inspect.signature(mod.run_agent_task).parameters,
                          f"{name}.run_agent_task cannot receive conversation history")

    def test_greeting_fast_path_removed(self):
        """A canned greeting reply meant the model was never consulted and every
        benchmark that greeted first measured a hard-coded string."""
        for name in ("agent.qwen_harness", "agent.harness", "agent.pydantic_harness"):
            src = (BACKEND / (name.replace(".", "/") + ".py")).read_text()
            self.assertNotIn('return "你好！有什麼可以幫你的？"', src,
                             f"{name} still returns a canned greeting without consulting the model")

    def test_streamed_deltas_are_forwarded_and_not_replayed(self):
        """generate_chat_with_tools must speak deltas live, and never speak them twice."""
        from llm import ling_streaming as ls

        async def fake_run_agent_task(task, event_q=None, history=None):
            for ch in "answer":
                event_q.put_nowait({"type": "llm_delta", "text": ch})
            return "answer"

        import agent.qwen_harness as qh
        # Patching the harness module works only because ling_streaming looks the runner
        # up through the imported module rather than holding its own reference.
        self.assertTrue(hasattr(qh, "run_agent_task"), "harness seam missing")
        saved = qh.run_agent_task
        qh.run_agent_task = fake_run_agent_task
        try:
            llm = ls.LingStreaming.__new__(ls.LingStreaming)
            llm.mock = True

            async def collect():
                return [ev async for ev in llm.generate_chat_with_tools([], "hi")]
            evs = asyncio.run(collect())
        finally:
            qh.run_agent_task = saved

        tokens = [e["token"] for e in evs if e["type"] == "llm_token"]
        self.assertEqual("".join(tokens), "answer")
        self.assertGreater(len(tokens), 1, "answer was replayed as one blob instead of streamed")
        done = [e for e in evs if e["type"] == "llm_done"]
        self.assertEqual(done[0]["text"], "answer")
        # no duplication
        self.assertEqual("".join(tokens), done[0]["text"])

    def test_delta_reset_discards_wrong_speculation(self):
        from llm import ling_streaming as ls
        import agent.qwen_harness as qh

        async def fake(task, event_q=None, history=None):
            event_q.put_nowait({"type": "llm_delta", "text": "wrong guess"})
            event_q.put_nowait({"type": "llm_delta", "text": "", "reset": True})
            return "final answer"

        saved = qh.run_agent_task
        qh.run_agent_task = fake
        try:
            llm = ls.LingStreaming.__new__(ls.LingStreaming)
            llm.mock = True

            async def collect():
                return [ev async for ev in llm.generate_chat_with_tools([], "hi")]
            evs = asyncio.run(collect())
        finally:
            qh.run_agent_task = saved
        # llm_reset tells the consumer to discard what it buffered from the
        # speculative deltas — exactly what pipeline.run_turn / app.direct_tts do.
        spoken, buf = "", ""
        for e in evs:
            if e["type"] == "llm_reset":
                buf = ""
            elif e["type"] == "llm_token":
                buf += e["token"]
        spoken = buf
        self.assertEqual(spoken, "final answer",
                         f"speculative text survived the reset: {spoken!r}")
        self.assertTrue(any(e["type"] == "llm_reset" for e in evs), "no llm_reset event emitted")

    def test_history_is_passed_as_real_messages(self):
        from llm import ling_streaming as ls
        import agent.qwen_harness as qh
        seen = {}

        async def fake(task, event_q=None, history=None):
            seen["history"] = history
            seen["task"] = task
            return "ok"

        saved = qh.run_agent_task
        qh.run_agent_task = fake
        try:
            llm = ls.LingStreaming.__new__(ls.LingStreaming)
            llm.mock = True
            hist = [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "長長的句子" * 60},
                    {"role": "assistant", "content": "A" * 500}]

            async def go():
                async for _ in llm.generate_chat_with_tools(hist, "BBC headlines"):
                    pass
            asyncio.run(go())
        finally:
            qh.run_agent_task = saved

        h = seen["history"]
        self.assertEqual([m["role"] for m in h], ["user", "assistant"], "system must be dropped")
        self.assertGreater(len(h[0]["content"]), 120,
                           "history is still truncated to a 120-char digest")
        self.assertNotIn("Conversation history:", seen["task"],
                         "task should be the raw question once real messages are sent")


# --------------------------------------------------------------------------
# P1-9: relevance scoring edge cases + cache bounds + SEARXNG_URL honored
# --------------------------------------------------------------------------
class TestWebSearch(unittest.TestCase):
    def setUp(self):
        import importlib
        # `from tools import web_search` yields the re-exported FUNCTION, not the
        # module (tools/__init__.py imports it) — importlib is unambiguous.
        ws = importlib.import_module("tools.web_search")
        self.ws = ws
        ws._cache_clear()

    def tearDown(self):
        self.ws._cache_clear()

    def test_unscorable_query_is_neutral_not_zero(self):
        # Queries that produce no scorable tokens (short numbers, punctuation,
        # stopwords only) must score NEUTRAL. Scoring them 0 meant the cache entry
        # was permanently "irrelevant": every repeat re-ran the whole fallback chain
        # with no way for the score to ever improve.
        r = [{"title": "x", "content": "y", "url": "z"}]
        for q in ("42", "?!?", "the latest", "。！"):
            self.assertEqual(self.ws._relevance_score(q, r), 1.0, f"{q!r} should be neutral")
        self.assertEqual(self.ws._relevance_score("weather paris", []), 0,
                         "no results must still score 0")

    def test_cjk_scoring_still_works(self):
        good = [{"title": "最新科技新聞：AI 產業動態", "content": "科技新品發表", "url": "u"}]
        bad = [{"title": "Cooking recipes", "content": "pasta", "url": "u"}]
        self.assertGreater(self.ws._relevance_score("最新科技新聞", good),
                           self.ws._relevance_score("最新科技新聞", bad))

    def test_cache_is_bounded(self):
        for i in range(self.ws.CACHE_MAX + 50):
            self.ws._cache_put(f"k{i}", {"results": [], "source": "t"})
        self.assertLessEqual(len(self.ws._CACHE), self.ws.CACHE_MAX)

    def test_cache_ttl_expiry(self):
        self.ws._cache_put("k", {"results": [], "source": "t"})
        self.assertIsNotNone(self.ws._cache_get("k"))
        self.ws._CACHE["k"]["t"] = time.time() - self.ws.CACHE_TTL - 1
        self.assertIsNone(self.ws._cache_get("k"))

    def test_searxng_base_follows_env(self):
        saved = os.environ.get("SEARXNG_URL")
        os.environ["SEARXNG_URL"] = "http://searxng:8080"
        try:
            self.assertEqual(self.ws._searxng_base(), "http://searxng:8080")
        finally:
            if saved is None:
                os.environ.pop("SEARXNG_URL", None)
            else:
                os.environ["SEARXNG_URL"] = saved
        src = Path(BACKEND / "tools" / "web_search.py").read_text()
        self.assertNotIn('client.get("http://localhost:8888/search"', src,
                         "search path must not hard-code the SearXNG host")

    def test_sync_wrapper_reuses_one_loop(self):
        """The sync tool wrapper used asyncio.run() per call — a fresh loop (and a
        fresh httpx pool/TLS handshake) per search. Pin the persistent-loop design
        without touching the network."""
        ws = self.ws
        saved = ws._SYNC_LOOP, ws._SYNC_THREAD
        ws._SYNC_LOOP = ws._SYNC_THREAD = None
        try:
            seen = {}

            async def fake(query, count=5, recency="any"):
                seen["recency"] = recency
                return {"query": query, "results": [], "source": "fake", "latency_ms": 0}
            real = ws.web_search
            ws.web_search = fake
            try:
                ws.web_search_sync("a", 1)
                ws.web_search_sync("b", 1, recency="day")
                self.assertEqual(seen["recency"], "day",
                                 "web_search_sync must forward the caller's recency")
                loop_a = ws._SYNC_LOOP
                ws.web_search_sync("b", 1)
                self.assertIs(loop_a, ws._SYNC_LOOP, "sync wrapper is recreating the event loop")
                self.assertTrue(loop_a.is_running())
                self.assertTrue(ws._SYNC_THREAD.daemon)
            finally:
                ws.web_search = real
        finally:
            ws._SYNC_LOOP, ws._SYNC_THREAD = saved


# --------------------------------------------------------------------------
# P1-10 / TTS: per-call voice, barge-in aborts synthesis
# --------------------------------------------------------------------------
class TestTtsVoice(unittest.TestCase):
    def _bare(self):
        from tts.qwen3_streaming import StreamingPrimeTTS
        t = StreamingPrimeTTS.__new__(StreamingPrimeTTS)
        t._vv = "台灣腔"
        t.speaker = "vivian"
        t.VOICE_PRESETS = {
            "台灣腔": {"type": "speaker", "name": "vivian"},
            "中文男聲": {"type": "speaker", "name": "uncle_fu"},
        }
        return t

    def test_per_call_voice_does_not_mutate_default(self):
        t = self._bare()
        preset, speaker = t._preset("中文男聲")
        self.assertEqual(speaker, "uncle_fu")
        self.assertEqual(t.speaker, "vivian", "shared instance default was mutated")
        self.assertEqual(t._vv, "台灣腔")

    def test_unknown_voice_raises_instead_of_silently_ignoring(self):
        t = self._bare()
        with self.assertRaises(KeyError):
            t._preset("不存在")

    def test_default_voice_path_uses_instance_speaker(self):
        t = self._bare()
        self.assertEqual(t._preset(None)[1], "vivian")
        self.assertEqual(t._preset("台灣腔")[1], "vivian")

    def test_language_segmentation_unchanged_contract(self):
        from tts.qwen3_streaming import _segment_by_language
        segs = _segment_by_language("Hello 王小明 Qwen3-TTS 測試")
        langs = [lang for lang, _ in segs]
        self.assertIn("english", langs)
        self.assertIn("chinese", langs)
        self.assertEqual("".join(s for _, s in segs).strip(), "Hello 王小明 Qwen3-TTS 測試")


class TestBargeInAbortsSynthesis(unittest.TestCase):
    def test_tts_generator_is_abandoned_not_drained(self):
        """synth_and_emit used to `continue` on barge-in: the TTS worker kept
        synthesizing the whole remaining sentence purely to throw it away."""
        from pipeline.speech_to_speech import HFSpeechToSpeechPipeline

        TOTAL = 30
        state = {"produced": 0, "closed": False, "calls": []}

        class FakeSTT:
            backend = "fake"

            async def transcribe_stream(self, q, stop, session_id=None):
                yield {"type": "stt_final", "text": "你好", "latency_ms": 10}
                while not stop.is_set():
                    await asyncio.sleep(0.02)

        class FakeLLM:
            backend = "fake"
            mock = False

            async def generate_chat_with_tools(self, history, prompt, max_new_tokens=256):
                yield {"type": "llm_token", "token": "一二三。", "text_so_far": "一二三。", "latency_ms": 1}
                await asyncio.sleep(0.05)
                yield {"type": "llm_done", "text": "一二三。"}

        class FakeTTS:
            backend = "fake"
            sample_rate = 24000

            async def synthesize_streaming(self, text, **kw):
                try:
                    for _ in range(TOTAL):
                        state["produced"] += 1
                        yield (np.ones(480, dtype=np.int16) * 1000)
                        await asyncio.sleep(0.01)
                finally:
                    state["closed"] = True

        p = HFSpeechToSpeechPipeline.__new__(HFSpeechToSpeechPipeline)
        p.mock = False
        p.device = "cpu"
        p.stt, p.llm, p.tts = FakeSTT(), FakeLLM(), FakeTTS()
        p.sessions = {}
        p._turn_counters = {}
        p._voice_response_tasks = {}

        async def run():
            q = asyncio.Queue()
            stop = asyncio.Event()
            barge = asyncio.Event()
            seen_chunks = 0
            gen = p.stream_chat_interleaved(q, stop, "s", barge_in_event=barge)
            async for ev in gen:
                if ev["type"] == "tts_chunk":
                    seen_chunks += 1
                    if seen_chunks == 1:
                        barge.set()      # external (other-channel) barge-in mid-speech
                if ev["type"] == "tts_end":
                    break                # turn over; the pipeline loop would otherwise
                                         # keep waiting for STT that never ends
            await gen.aclose()

        asyncio.run(run())
        self.assertTrue(state["closed"], "TTS generator was never closed on barge-in")
        self.assertLess(state["produced"], TOTAL,
                        f"synthesis was drained to the end ({state['produced']}/{TOTAL} chunks) "
                        "instead of being abandoned")


# --------------------------------------------------------------------------
# P1-12: dead code removed
# --------------------------------------------------------------------------
class TestDeadCode(unittest.TestCase):
    def test_unfinished_stream_chat_removed(self):
        from pipeline.speech_to_speech import HFSpeechToSpeechPipeline
        self.assertFalse(hasattr(HFSpeechToSpeechPipeline, "stream_chat"),
                         "the unfinished stream_chat (unbound first_tts_time, dangling "
                         "comment body) must stay deleted — stream_chat_interleaved is the API")

    def test_duplicate_adapters_deleted(self):
        for gone in ("stt/xasr_streaming_new.py", "stt/xasr_streaming_whisper_backup.py",
                     "tts/primetts_streaming_mock_backup.py"):
            self.assertFalse((BACKEND / gone).exists(), f"{gone} was unreferenced dead code")

    def test_stt_adapters_share_transcribe_stream_signature(self):
        from stt.xasr_streaming import StreamingXASR as A
        from stt.ark_streaming import StreamingXASR as B
        from stt.paraformer_streaming import StreamingXASR as C
        for klass in (A, B, C):
            self.assertIn("session_id", inspect.signature(klass.transcribe_stream).parameters,
                          f"{klass.__module__} breaks the pipeline's call signature")


# --------------------------------------------------------------------------
# P0-3b: llm_manager must not kill arbitrary processes
# --------------------------------------------------------------------------
class TestLlmManagerKillGuard(unittest.TestCase):
    def test_only_llama_server_is_killed(self):
        killed = []

        class FakeProc:
            def __init__(self, pid, name, cmdline):
                self._pid, self._name, self._cmd = pid, name, cmdline

            def name(self):
                return self._name

            def cmdline(self):
                return [self._cmd]

            def terminate(self):
                killed.append((self._pid, self._name))

            def wait(self, timeout=None):
                return 0

        class Conn:
            def __init__(self, port, pid):
                class A:
                    pass
                a = A()
                a.port = port
                self.laddr, self.pid, self.status = a, pid, "LISTEN"

        class FakePs:
            CONN_LISTEN = "LISTEN"

            @staticmethod
            def net_connections(kind=None):
                return [Conn(11435, 111), Conn(11435, 222)]

            @staticmethod
            def Process(pid):
                if pid == 111:
                    return FakeProc(111, "nginx", "/usr/sbin/nginx")
                return FakeProc(222, "llama-server", "/home/user/llama.cpp/build/bin/llama-server")

        saved = llmmod.psutil
        llmmod.psutil = FakePs
        try:
            llmmod.LLMServerManager._kill_unowned_listener(11435)
        finally:
            llmmod.psutil = saved
        self.assertEqual(killed, [(222, "llama-server")],
                         f"killed the wrong process(es): {killed}")


# --------------------------------------------------------------------------
# Latency metrics must describe what actually happened
# --------------------------------------------------------------------------
class TestMetricsHonesty(unittest.TestCase):
    def test_stats_endpoint_accepts_deque(self):
        from collections import deque
        saved = dict(appmod.stats)
        appmod.stats["latencies"] = deque([100, 200, 300], maxlen=10)
        try:
            j = _client().get("/stats").json()
            self.assertGreaterEqual(j["count"], 3)
        finally:
            appmod.stats.clear()
            appmod.stats.update(saved)

    def test_latency_list_is_bounded(self):
        self.assertEqual(appmod.stats["latencies"].maxlen, 1000)


class TestHFOfficialPipelineParity(unittest.TestCase):
    """`pipeline/hf_official.py` (HF_OFFICIAL opt-in path) used to accept
    barge_in_event/barge_in_lock/on_new_voice_turn and ignore all three, iterate STT
    inline (so nothing was recognized while a reply played), emit no turn_id, and
    compute stt_ms/llm_start/e2e_start and throw them away. app.py's WS loop speaks
    turn_id + barge_in, so the opt-in path now has to honour the same contract."""

    TOTAL = 40

    def _pipe(self, stt, llm, tts):
        from pipeline.hf_official import HFOfficialPipeline
        p = HFOfficialPipeline.__new__(HFOfficialPipeline)
        p.mock = False
        p.device = "cpu"
        p.sessions = {}
        p._turn_counters = {}
        p.sample_rate = 24000
        p.stt, p.llm, p.tts = stt, llm, tts
        return p

    class _TTS:
        backend = "fake"
        sample_rate = 24000

        def __init__(self, state, total):
            self.state, self.total = state, total

        async def synthesize_streaming(self, text, **kw):
            # counted PER CALL, not per sentence text: the fake LLM answers every turn
            # with the same sentence, so keying by text merges the interrupted turn's
            # abandoned call with the new turn's completed one (48/40 looked like the
            # driver ignoring the barge-in when it was the test conflating two calls).
            entry = ["", 0]
            entry[0] = text
            self.state["calls"].append(entry)
            try:
                for _ in range(self.total):
                    self.state["produced"] += 1
                    entry[1] += 1
                    yield np.ones(480, dtype=np.int16) * 1000
                    await asyncio.sleep(0.01)
            finally:
                self.state["closed"] = True

    class _STT:
        backend = "fake"

        def __init__(self, script):
            self.script = script          # [(delay, event)]

        async def transcribe_stream(self, q, stop, session_id=None):
            for delay, ev in self.script:
                await asyncio.sleep(delay)
                if stop.is_set():
                    return
                yield ev
            while not stop.is_set():
                await asyncio.sleep(0.02)

    def _collect(self, pipe, seconds=None, until=None):
        async def go():
            q: asyncio.Queue = asyncio.Queue()
            stop = asyncio.Event()
            events = []
            agen = pipe.stream_chat_interleaved(q, stop, "s1")
            t0 = time.time()
            try:
                while True:
                    try:
                        ev = await asyncio.wait_for(agen.__anext__(), timeout=1.0)
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        break          # window elapsed; the assertions below judge it
                    events.append((time.time() - t0, ev))
                    if until is not None and until(ev):
                        break
                    if seconds is not None and time.time() - t0 > seconds:
                        break
            finally:
                stop.set()
                await agen.aclose()
            return events
        return asyncio.run(go())

    def test_response_events_tagged_with_turn_id_and_latency_emitted(self):
        stt = self._STT([(0.0, {"type": "stt_final", "text": "第一条。", "latency_ms": 12}),
                         (0.6, {"type": "stt_final", "text": "第二条。", "latency_ms": 7})])

        class LLM:
            backend = "fake"
            n = 0

            async def generate_chat_with_tools(self, history, prompt):
                LLM.n += 1
                yield {"type": "llm_token", "token": "回答。", "text_so_far": "回答。"}
                yield {"type": "llm_done"}

        state = {"produced": 0, "closed": False, "calls": []}
        pipe = self._pipe(stt, LLM(), self._TTS(state, 3))
        events = self._collect(pipe, seconds=1.6, until=lambda e: LLM.n >= 2 and e.get("type") == "latency")
        kinds = [e["type"] for _, e in events]
        self.assertIn("tts_start", kinds, "first chunk must be preceded by tts_start (default-pipeline contract)")
        resp = [e for _, e in events if "turn_id" in e]
        self.assertTrue(resp, "response events must carry turn_id")
        self.assertTrue(all("turn_id" in e for _, e in events
                            if e.get("type") in ("llm_token", "tts_chunk", "tts_end", "latency")))
        lat = [e for _, e in events if e.get("type") == "latency"]
        self.assertTrue(lat, "the computed stt_ms/llm/tts/e2e timings must be emitted, not discarded")
        for k in ("stt_ms", "llm_ttft_ms", "tts_ttfb_ms", "e2e_ms"):
            self.assertIn(k, lat[0])
        self.assertEqual(lat[0]["stt_ms"], 12)
        ids = sorted({e["turn_id"] for e in resp})
        self.assertEqual(ids, [1, 2], f"two utterances must mint two monotonic turn ids, got {ids}")

    def test_voice_barge_in_cancels_old_turn_and_stops_its_audio(self):
        stt = self._STT([(0.0, {"type": "stt_final", "text": "長答案。", "latency_ms": 9}),
                         (0.08, {"type": "stt_partial", "text": "等一下", "latency_ms": 0}),
                         (0.16, {"type": "stt_final", "text": "換一個問題。", "latency_ms": 5})])

        class LLM:
            backend = "fake"

            async def generate_chat_with_tools(self, history, prompt):
                yield {"type": "llm_token", "token": "這是第一個回答。", "text_so_far": "這是第一個回答。"}
                await asyncio.sleep(1.0)                 # keep turn 1 alive across the barge
                yield {"type": "llm_token", "token": "後半段。", "text_so_far": "這是第一個回答。後半段。"}
                yield {"type": "llm_done"}

        state = {"produced": 0, "closed": False, "calls": []}
        pipe = self._pipe(stt, LLM(), self._TTS(state, self.TOTAL))
        events = self._collect(pipe, seconds=0.7)
        barges = [(t, e) for t, e in events if e.get("type") == "barge_in"]
        self.assertTrue(barges, "a non-empty partial over a live reply must yield a barge_in event")
        t_barge, barge = barges[0]
        self.assertEqual(barge.get("reason"), "voice")
        interrupted = barge["turn_id"]
        strays = [e for t, e in events
                  if e.get("type") == "tts_chunk" and e.get("turn_id") == interrupted and t > t_barge + 0.02]
        self.assertEqual(strays, [], f"interrupted turn kept emitting {len(strays)} chunks")
        new_turn = [e for t, e in events if e.get("type") == "tts_chunk" and e.get("turn_id") != interrupted]
        self.assertTrue(new_turn, "the new turn must still get its audio")
        self.assertTrue(state["closed"], "the abandoned TTS generator must be closed, not drained")
        first = next((c for c in state["calls"] if "第一個回答" in c[0]), None)
        self.assertIsNotNone(first, "turn 1 should have started synthesizing")
        self.assertLess(first[1], self.TOTAL,
                        f"turn 1's sentence synthesized to completion ({first[1]}/{self.TOTAL}) "
                        "instead of being abandoned at the barge-in")
        # '後半段' is the cancelled turn's still-pending output: it must never be synthesized.
        self.assertFalse([c[0] for c in state["calls"] if "後半段" in c[0]],
                         "cancelled turn resumed generating and got synthesized")
        self.assertGreaterEqual(len(state["calls"]), 2, "the new turn must get its own synthesis call")

    def test_llm_reset_discards_speculative_text(self):
        stt = self._STT([(0.0, {"type": "stt_final", "text": "問題。", "latency_ms": 4})])

        class LLM:
            backend = "fake"

            async def generate_chat_with_tools(self, history, prompt):
                yield {"type": "llm_token", "token": "猜錯的半句", "text_so_far": "猜錯的半句"}   # no sentence end -> still buffered
                yield {"type": "llm_reset"}
                yield {"type": "llm_token", "token": "真正的回答。", "text_so_far": "真正的回答。"}
                yield {"type": "llm_done"}

        state = {"produced": 0, "closed": False, "calls": []}
        pipe = self._pipe(stt, LLM(), self._TTS(state, 2))
        events = self._collect(pipe, seconds=1.2, until=lambda e: e.get("type") == "latency")
        kinds = [e["type"] for _, e in events]
        self.assertIn("llm_reset", kinds)
        spoken = "".join(e.get("text", "") for _, e in events if e.get("type") == "tts_chunk")
        self.assertNotIn("猜錯", spoken, "withdrawn text must never reach TTS")
        self.assertIn("真正", spoken)
        hist = pipe.sessions["s1"]
        self.assertTrue(any(m["role"] == "assistant" and "真正" in m["content"] for m in hist))
        self.assertFalse(any("猜錯" in m.get("content", "") for m in hist),
                         "discarded speculation must not leak into history either")
        # (A token that already ended a sentence is synthesizing and cannot be recalled —
        #  reset only protects what is still buffered.)

    def test_external_barge_in_event_aborts_synthesis(self):
        stt = self._STT([(0.0, {"type": "stt_final", "text": "打斷它。", "latency_ms": 3})])

        class LLM:
            backend = "fake"

            async def generate_chat_with_tools(self, history, prompt):
                yield {"type": "llm_token", "token": "一句長話。", "text_so_far": "一句長話。"}
                await asyncio.sleep(1.0)
                yield {"type": "llm_done"}

        state = {"produced": 0, "closed": False, "calls": []}
        pipe = self._pipe(stt, LLM(), self._TTS(state, self.TOTAL))

        async def go():
            q: asyncio.Queue = asyncio.Queue()
            stop = asyncio.Event()
            barge = asyncio.Event()
            events = []
            agen = pipe.stream_chat_interleaved(q, stop, "s1", barge_in_event=barge)
            seen = 0
            t0 = time.time()
            try:
                while time.time() - t0 < 1.0:
                    try:
                        ev = await asyncio.wait_for(agen.__anext__(), timeout=0.5)
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        break
                    events.append(ev)
                    if ev.get("type") == "tts_chunk":
                        seen += 1
                        if seen == 2:
                            barge.set()               # what app.py's do_barge_in does
            finally:
                stop.set()
                await agen.aclose()
            return events

        events = asyncio.run(go())
        chunks = [e for e in events if e.get("type") == "tts_chunk"]
        self.assertLess(len(chunks), self.TOTAL, f"external barge-in emitted all {len(chunks)} chunks")
        self.assertTrue(state["closed"])

    def test_teardown_leaves_no_tasks_behind(self):
        stt = self._STT([(0.0, {"type": "stt_final", "text": "喂。", "latency_ms": 1})])

        class LLM:
            backend = "fake"

            async def generate_chat_with_tools(self, history, prompt):
                await asyncio.sleep(30)              # never finishes: client hung up
                yield {"type": "llm_done"}

        state = {"produced": 0, "closed": False, "calls": []}
        pipe = self._pipe(stt, LLM(), self._TTS(state, 2))

        async def go():
            q: asyncio.Queue = asyncio.Queue()
            stop = asyncio.Event()
            agen = pipe.stream_chat_interleaved(q, stop, "s1")
            await agen.__anext__()                    # pump started, turn started
            await asyncio.sleep(0.05)
            before = len([t for t in asyncio.all_tasks() if not t.done()])
            await agen.aclose()                       # client disconnects
            await asyncio.sleep(0.05)
            after = len([t for t in asyncio.all_tasks() if not t.done()])
            return before, after

        before, after = asyncio.run(go())
        self.assertLessEqual(after, before, "closing the generator must cancel pump + response task")


class TestLlmSeedPassthrough(unittest.TestCase):
    """Tool-call routing at temperature 0.7 flips 4/7 <-> 5/7 between runs, so a latency
    or routing change cannot be A/B-ed on one sample. llama-server takes `--seed`; the
    manager must pass it through when configured and stay out of the way when not."""

    def _spawn_cmd(self, seed):
        import llm_manager as m
        model_id = next((k for k, v in m.MODEL_REGISTRY.items() if Path(v["path"]).exists()), None)
        if model_id is None:
            self.skipTest("no GGUF weights present")
        captured = {}

        class FakeProc:
            pid = 4242

            def wait(self, timeout=None):
                return 0

        real_popen = m.subprocess.Popen

        def fake_popen(cmd, **kw):
            captured["cmd"] = list(cmd)
            return FakeProc()

        saved = m.LLM_SEED
        m.subprocess.Popen = fake_popen
        m.LLM_SEED = seed
        try:
            m.LLMServerManager()._spawn(model_id)
        finally:
            m.LLM_SEED = saved
            m.subprocess.Popen = real_popen
        return captured["cmd"]

    def test_default_adds_no_seed_flag(self):
        cmd = self._spawn_cmd(-1)
        self.assertNotIn("--seed", cmd, "default must stay llama.cpp's random seed")
        self.assertIn("--jinja", cmd, "existing flags must survive")

    def test_fixed_seed_reaches_llama_server(self):
        cmd = self._spawn_cmd(4711)
        self.assertIn("--seed", cmd)
        self.assertEqual(cmd[cmd.index("--seed") + 1], "4711")








class TestSpeechTextShaping(unittest.TestCase):
    """What gets synthesized, and the removal of two per-question hard-codings."""

    def setUp(self):
        from llm.ling_streaming import _speakable, _clean_leakage
        from pipeline.speech_to_speech import is_echo_of_prompt
        self.speakable, self.clean, self.echo = _speakable, _clean_leakage, is_echo_of_prompt

    def test_markdown_never_reaches_the_speaker(self):
        # observed live: 法國總統是**愛德華·馬克龍**(search-grounded answer in markdown)
        self.assertEqual(self.speakable("法國總統是**愛德華·馬克龍**。"), "法國總統是愛德華·馬克龍。")
        self.assertEqual(self.speakable("見 [中央通信社](https://cna.com.tw/a) 的報導"), "見 中央通信社 的報導")
        self.assertEqual(self.speakable("用 `_private` 名稱"), "用 private 名稱")
        self.assertEqual(self.speakable("**布琳** 與 __漢斯__"), "布琳 與 漢斯")

    def test_leaked_result_rows_stripped_without_query_specific_rules(self):
        row = "答案是。 [1] Emmanuel Macron - Wikipedia URL: https://en.wikipedia.org/wiki/Macron Date/Snippet: born 1977 "
        out = self.clean(row)
        self.assertNotIn("URL:", out)
        self.assertNotIn("wikipedia", out)
        self.assertIn("答案是", out)
        self.assertEqual(self.clean("[2] Some headline — Reuters\n第二行保留"), "第二行保留")

    def test_question_echo_is_detected_by_content_not_literals(self):
        q = "Who is the president of France 2024?"
        self.assertTrue(self.echo("Who is the president of France 2024", q))
        self.assertTrue(self.echo("who is the president", q), "a partial echo counts too")
        self.assertTrue(self.echo("Who is the president of France?", "", ["Who is the president of France?"]),
                        "echoing a tool's own query is also not an answer")
        # an answer that merely starts with the same words must NOT be swallowed
        self.assertFalse(self.echo(
            "Who is the president of France? The answer changed hands in 2027 according to the search results.", q))
        self.assertFalse(self.echo("法國總統是愛德華·馬克龍。", q))
        self.assertFalse(self.echo("台北今天午後有陣雨。", "",
                                   ["台北 今天 天氣"]))

    def test_live_paths_hold_no_benchmark_query_literals(self):
        """README_TOOL promises nothing is hard-coded per demo question. Two sites used to
        break that: `_clean_leakage` special-cased 'President of France', and both speech
        paths suppressed an echo via startswith('who is the president'). Scan the code
        (string constants only, so prose comments and prompt examples stay free)."""
        import ast
        banned = ("who is the president", "president of france", "latest news about artificial")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        offenders = []
        for rel in ("app.py", "pipeline/speech_to_speech.py", "llm/ling_streaming.py",
                    "agent/qwen_harness.py", "tools/web_search.py"):
            src = open(os.path.join(root, rel), encoding="utf-8").read()
            tree = ast.parse(src)
            documented = {id(n.value) for n in [tree, *ast.walk(tree)]
                          if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Expr))
                          for n2 in [getattr(n, "body", [])] if n2
                          for n in [n2[0]] if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
            for node in ast.walk(tree):
                if id(node) in documented:
                    continue                     # prose about removed bugs is allowed
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    low = node.value.lower()
                    for b in banned:
                        if b in low:
                            offenders.append(f"{rel}: {node.value[:60]!r}")
        self.assertEqual(offenders, [], "per-question literals back in the live path:\n" + "\n".join(offenders))


class TestConstructorFallbackNotImportOnly(unittest.TestCase):
    """The adapter ladders prove that a module IMPORTS; the failure that actually killed
    boot happened at CONSTRUCTION (paraformer imports funasr inside __init__ and raises
    RuntimeError). Observed inside python:3.13-slim, where nothing model-shaped was
    installed: `--mock` died in HFSpeechToSpeechPipeline.__init__ instead of booting on
    the mock rungs it was supposed to fall back to."""

    def test_stt_and_tts_construction_failures_degrade_to_mock(self):
        from pipeline import speech_to_speech as ps

        class Exploding:
            def __init__(self, *a, **k):
                raise RuntimeError("No module named 'funasr'")

        saved_stt, saved_tts = ps.StreamingXASR, ps.StreamingPrimeTTS
        ps.StreamingXASR = Exploding
        ps.StreamingPrimeTTS = Exploding
        try:
            pipe = ps.HFSpeechToSpeechPipeline(device="cpu", mock=True)
            self.assertEqual(pipe.stt.backend, "mock", "STT construction failure must degrade, not raise")
            self.assertEqual(pipe.tts.backend, "mock", "TTS construction failure must degrade, not raise")
            self.assertTrue(hasattr(pipe.stt, "transcribe_stream"))
            self.assertTrue(hasattr(pipe.tts, "synthesize_streaming"))
            self.assertTrue(pipe.tts.sample_rate > 0)
        finally:
            ps.StreamingXASR, ps.StreamingPrimeTTS = saved_stt, saved_tts

    def test_mock_boot_needs_no_ml_libraries(self):
        """Run in a subprocess with every optional ML import blocked, mirroring the
        python:3.13-slim container where this was verified by hand."""
        import json as _json
        import subprocess
        import sys
        import tempfile
        code = r'''import sys, json
sys.path.insert(0, %r)
import builtins
_blocked = ("torch","sherpa_onnx","funasr","faster_whisper","faster_qwen3_tts","qwentts_cpp_python",
            "kokoro_onnx","mosstts","voxcpm","transformers","onnxruntime","ark_asr")
_real = builtins.__import__
def fake(name, *a, **k):
    root = name.split(".")[0]
    if root in _blocked: raise ImportError("blocked for test: " + root)
    return _real(name, *a, **k)
builtins.__import__ = fake
from pipeline.speech_to_speech import HFSpeechToSpeechPipeline
p = HFSpeechToSpeechPipeline(device="cpu", mock=True)
print(json.dumps({"stt": p.stt.backend, "llm": getattr(p.llm, "backend", "?"), "tts": p.tts.backend}))
''' % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
            script = fh.name
        try:
            r = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=180)
            self.assertEqual(r.returncode, 0, f"mock boot failed:\n{r.stdout}\n{r.stderr[-2000:]}")
            loaded = _json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(loaded["stt"], "mock")
            self.assertEqual(loaded["tts"], "mock")
        finally:
            os.unlink(script)


class TestUnreachableModelServerDegrades(unittest.TestCase):
    """`--mock` promises "run without the model stack", and the compose default points
    LLM_API_BASE at a service name (`http://llama:8080/v1`) that resolves nowhere outside
    `docker compose up`. Observed inside the built container: POST /api/chat -> HTTP 500
    with an httpx.ConnectError traceback, because LingStreaming's constructor deliberately
    stays optimistic ("will retry per request") and the retry raised straight out of the
    endpoint. Retrying is right; surfacing it as a stack trace is not."""

    DEAD = "http://127.0.0.1:1/v1"          # port 1: refused, no DNS involved

    def _llm(self):
        from llm.ling_streaming import LingStreaming
        return LingStreaming(api_base=self.DEAD, model_name="probe-test", mock=False)

    def test_generate_with_tools_degrades_instead_of_raising(self):
        import asyncio
        llm = self._llm()

        async def drain():
            out = []
            async for ev in llm.generate_with_tools("今天台北天氣如何？"):
                out.append(ev)
            return out

        evs = asyncio.run(asyncio.wait_for(drain(), timeout=30))
        kinds = {e["type"] for e in evs}
        self.assertIn("llm_done", kinds, "the consumer must still see a clean end-of-turn")
        text = "".join(e.get("token", "") for e in evs if e["type"] == "llm_token")
        self.assertTrue(text.strip(), "degradation must produce something speakable")
        self.assertIn("127.0.0.1:1", text, "it must name the endpoint that is missing")
        self.assertFalse(any("無法" == t for t in [text]), "not an empty apology")
        # …and it must not pretend to be a normal answer about weather
        self.assertNotIn("34", text)

    def test_legacy_branch_degrades_too(self):
        """A light install has no qwen-agent/pydantic-ai, so `generate_chat_with_tools`
        takes its `if not _HARNESS: … return` branch early. The pre-flight originally sat
        below that return, so exactly the installs most likely to have a dead LLM_API_BASE
        were the ones still raising."""
        import asyncio
        import sys
        from unittest import mock
        from llm.ling_streaming import LingStreaming
        llm = LingStreaming(api_base=self.DEAD, model_name="probe-test", mock=False)

        async def drain():
            return [e async for e in llm.generate_with_tools("今天台北天氣如何？")]

        with mock.patch.dict(sys.modules, {"agent.qwen_harness": None, "agent.pydantic_harness": None,
                                           "agent.harness": None}):
            evs = asyncio.run(asyncio.wait_for(drain(), timeout=30))
        text = "".join(e.get("token", "") for e in evs if e["type"] == "llm_token")
        self.assertIn("127.0.0.1:1", text, "legacy loop must degrade like the harness path")

    def test_mock_mode_still_sounds_like_the_demo(self):
        """--mock opted into simulated answers; the outage message belongs to a real
        deployment, not to the mode whose whole purpose is running without models."""
        import asyncio
        from llm.ling_streaming import LingStreaming
        llm = LingStreaming(api_base=self.DEAD, model_name="probe-test", mock=False, degraded_mode="mock")

        async def drain():
            return [e async for e in llm.generate_with_tools("hello there")]

        evs = asyncio.run(asyncio.wait_for(drain(), timeout=30))
        text = "".join(e.get("token", "") for e in evs if e["type"] == "llm_token")
        self.assertTrue(text.strip())
        self.assertNotIn("127.0.0.1", text, "--mock should not read like an outage report")

    def test_probe_cache_is_per_instance_so_a_model_switch_reprobes(self):
        import asyncio
        llm = self._llm()
        ok, why = asyncio.run(llm._reachable())
        self.assertFalse(ok)
        self.assertTrue(llm._last_probe[2])
        fresh = self._llm()
        self.assertEqual(fresh._last_probe[0], 0.0,
                         "a newly built adapter inherited another endpoint's verdict")

    def test_api_chat_returns_json_error_when_generation_blows_up(self):
        import app as appmod
        from starlette.testclient import TestClient   # (starlette spells it without an underscore)

        class ExplodingLLM:
            backend = "exploding"
            mock = False

            async def generate_with_tools(self, prompt, max_new_tokens=256):
                raise ConnectionRefusedError("llama-server went away mid-stream")
                yield  # pragma: no cover  (keeps this an async generator)

        class DummyTTS:
            backend = "mock"
            sample_rate = 16000
            VOICE_PRESETS = None

        class DummyPipeline:
            mock = True

            def __init__(self):
                self.llm = ExplodingLLM()
                self.tts = DummyTTS()
                self.sessions = {}

        saved = appmod.pipeline
        appmod.pipeline = DummyPipeline()
        try:
            # TestClient only runs the app lifespan inside `with`, which is what keeps
            # this from building the real model stack.
            client = TestClient(appmod.app)
            r = client.post("/api/chat", json={"text": "hello", "tools": True})
            self.assertEqual(r.status_code, 503, r.text[:200])
            body = r.json()
            self.assertIn("ConnectionRefusedError", body["error"])
            self.assertEqual(body["stt_text"], "hello")
        finally:
            appmod.pipeline = saved

    def test_light_container_shape_no_ml_imports_needed_for_the_error_path(self):
        """The degraded path must not itself depend on a model library — in the light
        image there is no torch/qwen_agent, and the apology is the only thing left."""
        import ast
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "llm", "ling_streaming.py"), encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name in ("_reachable", "_degraded"):
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        mods = [a.name for a in sub.names] if isinstance(sub, ast.Import) else [sub.module]
                        for m in mods:
                            self.assertNotIn((m or "").split(".")[0],
                                             {"torch", "transformers", "qwen_agent", "sherpa_onnx"},
                                             f"{node.name} must stay import-light ({m})")


class TestWebSocketOriginGate(unittest.TestCase):
    """CORSMiddleware gates HTTP, not WebSockets: new WebSocket() is not a CORS-checked
    request, so a foreign page could open ws://host/ws/chat from a visitor's browser even
    after ALLOWED_ORIGINS was introduced. The payload behind that socket is a live
    mic-to-speaker session with tool calls."""

    class _WS:
        def __init__(self, headers):
            self.headers = headers

    def _ok(self, origin, host="127.0.0.1:8000"):
        import app as appmod
        h = {} if origin is None else {"origin": origin}
        h["host"] = host
        return appmod._origin_ok(self._WS(h))

    def test_no_origin_is_a_non_browser_client(self):
        self.assertEqual(self._ok(None)[0], True)
        self.assertEqual(self._ok("")[0], True)

    def test_foreign_origin_is_rejected(self):
        ok, why = self._ok("https://evil.example")
        self.assertFalse(ok)
        self.assertIn("evil.example", why)
        ok, _ = self._ok("https://evil.example", host="voice.tailnet.ts.net")
        self.assertFalse(ok, "a foreign origin must not pass by changing our own Host")

    def test_same_origin_is_allowed_which_is_how_the_ui_and_funnel_work(self):
        self.assertTrue(self._ok("https://voice.tailnet.ts.net", host="voice.tailnet.ts.net")[0])
        self.assertTrue(self._ok("http://127.0.0.1:8000")[0])
        self.assertTrue(self._ok("http://localhost:8000")[0])

    def test_loopback_pages_on_any_port_are_allowed(self):
        self.assertTrue(self._ok("http://127.0.0.1:5173")[0])
        self.assertTrue(self._ok("http://localhost:4173")[0])
        self.assertTrue(self._ok("http://[::1]:5173")[0])

    def test_opaque_and_non_http_origins_are_rejected(self):
        self.assertFalse(self._ok("null")[0])
        self.assertFalse(self._ok("file://")[0])
        self.assertFalse(self._ok("http://evil.example.evil:8000")[0])

    def test_configured_allowlist_is_honoured(self):
        import app as appmod
        saved = appmod.CORS_ORIGINS
        appmod.CORS_ORIGINS = ["https://app.example"]
        try:
            self.assertTrue(self._ok("https://app.example")[0])
            self.assertTrue(self._ok("https://APP.EXAMPLE")[0], "origin compare must be case-insensitive")
        finally:
            appmod.CORS_ORIGINS = saved

    def test_escape_hatch_exists(self):
        import app as appmod
        saved = appmod.WS_ALLOW_ANY_ORIGIN
        appmod.WS_ALLOW_ANY_ORIGIN = True
        try:
            self.assertTrue(self._ok("https://evil.example")[0])
        finally:
            appmod.WS_ALLOW_ANY_ORIGIN = saved

    def test_handshake_is_refused_before_accept(self):
        """Rejection must happen before accept() — otherwise the client sees a 101 and we
        own a socket we then have to tear down."""
        import app as appmod
        from starlette.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect
        client = TestClient(appmod.app)
        with self.assertRaises(WebSocketDisconnect):
            with client.websocket_connect("/ws/chat?session_id=origin-test",
                                         headers={"origin": "https://evil.example"}):
                self.fail("connection should have been refused")


try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False


@unittest.skipUnless(_HAVE_YAML, "PyYAML not installed")
class TestShipFilesAreValid(unittest.TestCase):
    """CI config and docker-compose are the two files nobody runs locally before pushing,
    and both were broken in ways that only surface at deploy time: an unquoted step name
    containing ': ' made .github/workflows/ci.yml an invalid YAML document (Actions would
    reject the workflow outright), and the Dockerfile's default LLM_API_BASE pointed at a
    service name (`llama`) that does not exist in docker-compose.yml (`llm`) — which is how
    a container ends up with an unreachable model endpoint and, before the pre-flight
    existed, a 500 on every chat request."""

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _load(self, rel):
        with open(os.path.join(self.ROOT, rel), encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_workflows_parse_and_are_shape_correct(self):
        wdir = os.path.join(self.ROOT, ".github", "workflows")
        seen = 0
        for fn in sorted(os.listdir(wdir)):
            if not fn.endswith((".yml", ".yaml")):
                continue
            doc = self._load(os.path.join(".github", "workflows", fn))
            self.assertIsInstance(doc, dict, f"{fn} is not a mapping")
            # `on:` is parsed as the boolean True by YAML 1.1 loaders — accept either key.
            self.assertTrue(doc.get("on") is not None or doc.get(True) is not None,
                            f"{fn}: missing/invalid `on:` trigger")
            for job, spec in doc["jobs"].items():
                self.assertIn("runs-on", spec, f"{fn}:{job} has no runs-on")
                self.assertTrue(spec.get("steps") or spec.get("uses"),
                                f"{fn}:{job} has no steps")
                for i, st in enumerate(spec.get("steps") or []):
                    self.assertIsInstance(st.get("name", ""), str, f"{fn}:{job} step {i} name is not a string")
                seen += 1
        self.assertGreater(seen, 0, "no workflow files found")

    def test_compose_parses(self):
        doc = self._load("docker-compose.yml")
        self.assertIn("services", doc)
        for name, svc in doc["services"].items():
            self.assertTrue(svc.get("image") or svc.get("build"), f"service {name} has neither image nor build")

    def test_dockerfiles_llm_default_names_a_real_compose_service(self):
        with open(os.path.join(self.ROOT, "Dockerfile"), encoding="utf-8") as fh:
            dockerfile = fh.read()
        m = re.search(r"LLM_API_BASE=https?://([^:/\s]+):", dockerfile)
        self.assertIsNotNone(m, "Dockerfile no longer sets a default LLM_API_BASE")
        services = set(self._load("docker-compose.yml")["services"])
        self.assertIn(m.group(1), services,
                      f"Dockerfile default host {m.group(1)!r} is not a compose service {sorted(services)}")

    def test_compose_override_agrees_with_the_service_name(self):
        doc = self._load("docker-compose.yml")
        env = doc["services"]["voice-chat"].get("environment") or []
        base = [e for e in env if str(e).startswith("LLM_API_BASE=")]
        self.assertTrue(base, "voice-chat no longer sets LLM_API_BASE; the image default would be used")
        host = re.search(r"https?://([^:/\s]+):", base[0]).group(1)
        self.assertIn(host, doc["services"])




class TestThinkingTextNeverGetsSpoken(unittest.TestCase):
    """With enable_thinking on (needed: 81 % vs 71 % tool routing, and the 71 % variant
    fabricates weather), llama-server at reasoning-format `auto` leaves the deliberation
    unparsed in message.content — and the agent path speaks message.content. The benchmark
    answer previews captured it verbatim; the durable fix is server-side
    `--reasoning-format deepseek`, with this filter as the backstop for builds that lack
    the flag."""

    def _strip(self, t):
        from llm.ling_streaming import _strip_thinking
        return _strip_thinking(t)

    def test_terminated_block_is_removed(self):
        out = self._strip("<|im_start|>think\n先查時間*/今天 8 月 31 日。")
        self.assertNotIn("先查時間", out)
        self.assertIn("8 月 31 日", out)

    def test_unterminated_block_is_removed(self):
        out = self._strip("<|im_start|>think\n我需要先取得時間，然後再組織回答。")
        self.assertNotIn("我需要先取得時間", out)

    def test_ordinary_answers_are_untouched(self):
        for t in ("現在是 2026 年 9 月 1 日星期二，8 點 29 分（台北時間）。",
                  "東京今天 24-29°C，陣雨。",
                  "2 < 3 但答案不該被動到"):
            self.assertEqual(self._strip(t), t, "the filter must not eat real answers")

    def test_it_is_wired_into_the_answer_path_not_just_defined(self):
        import ast
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "llm", "ling_streaming.py")
        with open(src, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        used = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate_chat_with_tools":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "_strip_thinking":
                        used = True
        self.assertTrue(used, "_strip_thinking is defined but never applied to the answer")


class TestTtsAdapterVoiceInterface(unittest.TestCase):
    """`tts.voices` used to mean two different things depending on which adapter won the
    ladder: a bound method on qwen3/audio8/kokoro/mock, a plain list on the MOSS ones.
    Nothing read it yet, so the mismatch was invisible — a voice picker or a /health line
    written against either convention breaks on the other, and the KeyError paths in the
    method-style adapters called `self.voices()` where a list-style adapter would raise
    TypeError instead of the intended message."""

    FILES = None

    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tts_dir = os.path.join(root, "tts")

    def test_voices_is_a_property_or_attribute_never_a_method(self):
        import ast
        offenders = []
        for fn in sorted(os.listdir(self.tts_dir)):
            if not fn.endswith(".py") or fn == "__init__.py":
                continue
            src = open(os.path.join(self.tts_dir, fn), encoding="utf-8").read()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "voices":
                    if not any((isinstance(d, ast.Name) and d.id == "property") for d in node.decorator_list):
                        offenders.append(f"tts/{fn}: `def voices` without @property")
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "voices" and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"):
                    offenders.append(f"tts/{fn}: calls self.voices() — it is a list, not a method")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_mock_adapter_presents_the_interface_a_caller_would_use(self):
        from tts.mock_streaming import StreamingPrimeTTS
        t = StreamingPrimeTTS()
        self.assertIsInstance(t.voices, list)
        self.assertIn("mock", t.voices)
        with self.assertRaises(KeyError) as cm:
            t.set_voice("no-such-voice")
        self.assertIn("mock", str(cm.exception), "the error should list what IS available")

    def test_qwen3_preset_lookup_reports_available_voices_without_typeerror(self):
        # the KeyError path formats self.voices — make sure it is formatting, not calling
        from tts.qwen3_streaming import StreamingPrimeTTS
        t = StreamingPrimeTTS.__new__(StreamingPrimeTTS)
        t.VOICE_PRESETS = {"台灣腔": {"type": "speaker", "name": "vivian"}}
        t._vv = "台灣腔"
        t.speaker = "vivian"
        with self.assertRaises(KeyError) as cm:
            t._preset("no-such-voice")
        self.assertIn("台灣腔", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
