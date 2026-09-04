"""Unit tests for POST/GET /v1/llm-config (OpenCode Go key + model), in isolation
from the realtime pipeline: _install_llm_config_route only needs a bare FastAPI
app, so the whole speech_to_speech server factory does not have to run."""
import os
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import s2s.serve as serve


def _app():
    app = FastAPI()
    serve._install_llm_config_route(app)
    return TestClient(app)


class TestLlmConfigRoute(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for k in ("LLM_API_BASE", "LLM_MODEL_ID", "LLM_API_KEY"):
            os.environ.pop(k, None)
        serve._llm_stage = None

    def tearDown(self):
        self._env.stop()

    def test_get_reports_no_key_set_by_default(self):
        r = _app().get("/v1/llm-config")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["provider"], "opencode-go")
        self.assertFalse(body["key_set"])
        self.assertEqual(body["key_hint"], "")
        self.assertIn("mimo-v2.5", body["models"])

    def test_post_without_a_key_is_rejected(self):
        r = _app().post("/v1/llm-config", json={"model": "mimo-v2.5"})
        self.assertEqual(r.status_code, 400)

    def test_post_accepts_a_model_not_in_the_fixed_fallback_list(self):
        # Not checked against a catalogue: the base URL is fixed regardless of
        # model, so an unrecognised id can only ever fail at OpenCode Go itself.
        r = _app().post("/v1/llm-config", json={"api_key": "sk-test", "model": "some-new-model"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["model"], "some-new-model")

    def test_post_sets_the_endpoint_env_and_masks_the_key_on_readback(self):
        client = _app()
        r = client.post("/v1/llm-config", json={"api_key": "sk-abcdefghij", "model": "mimo-v2.5-pro"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["key_set"])
        self.assertEqual(body["model"], "mimo-v2.5-pro")
        self.assertNotIn("sk-abcdefghij", str(body))
        self.assertEqual(os.environ["LLM_API_BASE"], serve.OPENCODE_GO_BASE)
        self.assertEqual(os.environ["LLM_MODEL_ID"], "mimo-v2.5-pro")
        self.assertEqual(os.environ["LLM_API_KEY"], "sk-abcdefghij")

    def test_post_defaults_to_the_first_model_when_none_is_given(self):
        client = _app()
        r = client.post("/v1/llm-config", json={"api_key": "sk-test"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["model"], serve.OPENCODE_GO_MODELS[0])

    def test_post_reconfigures_the_live_stage_when_one_is_running(self):
        stage = mock.MagicMock()
        serve._llm_stage = stage
        _app().post("/v1/llm-config", json={"api_key": "sk-test", "model": "mimo-v2.5"})
        stage.reconfigure.assert_called_once_with(serve.OPENCODE_GO_BASE, "mimo-v2.5", "sk-test")


class TestLlmModelsRoute(unittest.TestCase):
    """GET /v1/llm-models: never touches the network for GET /v1/llm-config
    itself (see _MODEL_CACHE's docstring) -- these mock httpx so the catalogue
    fetch this route DOES make stays offline-safe in the test suite too."""

    def setUp(self):
        serve._MODEL_CACHE[:] = [0.0, []]

    def test_returns_the_live_catalogue(self):
        response = mock.MagicMock()
        response.json.return_value = {"data": [{"id": "mimo-v2.5"}, {"id": "kimi-k3"}]}
        with mock.patch("httpx.get", return_value=response) as get:
            r = _app().get("/v1/llm-models")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sorted(r.json()["models"]), ["kimi-k3", "mimo-v2.5"])
        get.assert_called_once()

    def test_a_cached_result_is_not_refetched(self):
        response = mock.MagicMock()
        response.json.return_value = {"data": [{"id": "mimo-v2.5"}]}
        with mock.patch("httpx.get", return_value=response) as get:
            _app().get("/v1/llm-models")
            _app().get("/v1/llm-models")
        get.assert_called_once()

    def test_a_fetch_failure_falls_back_to_the_fixed_list(self):
        with mock.patch("httpx.get", side_effect=ConnectionError("offline")):
            r = _app().get("/v1/llm-models")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["models"], list(serve.OPENCODE_GO_MODELS))


if __name__ == "__main__":
    unittest.main()
