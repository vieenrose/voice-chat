"""A refused turn should say which refusal it was.

Once the LLM can be a hosted provider, refusals are routine and each needs a
different action: a bad key, an empty balance, a rate-limited free model. One
generic apology sent them all to the same dead end.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.qwen_harness import _provider_failure_zh


class _Coded(Exception):
    def __init__(self, code):
        super().__init__(f"provider said {code}")
        self.status_code = code


class TestProviderFailureMessages(unittest.TestCase):
    def test_each_cause_is_distinguishable(self):
        seen = {}
        for code in (401, 402, 403, 404, 429):
            msg = _provider_failure_zh(_Coded(code))
            self.assertNotIn(code, seen.values())
            seen[msg] = code
        self.assertEqual(len(seen), 5, "each cause needs its own message")

    def test_key_and_quota_name_the_remedy(self):
        self.assertIn("金鑰", _provider_failure_zh(_Coded(401)))
        self.assertIn("額度", _provider_failure_zh(_Coded(402)))
        self.assertIn("模型", _provider_failure_zh(_Coded(429)))

    def test_status_is_recovered_from_the_message_when_not_an_attribute(self):
        # openai/qwen-agent wrap the original error, so the code is often only in text
        msg = _provider_failure_zh(Exception("Error code: 429 - rate-limited upstream"))
        self.assertIn("限流", msg)

    def test_server_errors_are_grouped(self):
        for code in (500, 502, 503):
            self.assertIn("供應者", _provider_failure_zh(_Coded(code)))

    def test_unknown_failures_stay_generic_and_never_leak_the_exception(self):
        exc = Exception("httpx.ConnectError: [Errno 111] to 10.1.2.3:8080")
        msg = _provider_failure_zh(exc)
        self.assertNotIn("10.1.2.3", msg)
        self.assertNotIn("Errno", msg)
        self.assertTrue(msg.startswith("抱歉"))


if __name__ == "__main__":
    unittest.main()
