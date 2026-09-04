import unittest

from s2s import turn_trace


class TestTurnTrace(unittest.TestCase):
    def setUp(self):
        turn_trace.begin("t1")

    def test_begin_resets_reasoning_steps_and_usage(self):
        turn_trace.add_reasoning("stale")
        turn_trace.begin("t2")
        snap = turn_trace.snapshot()
        self.assertEqual(snap["turn_id"], "t2")
        self.assertEqual(snap["reasoning"], "")
        self.assertEqual(snap["steps"], [])
        self.assertIsNone(snap["usage"])
        self.assertFalse(snap["done"])

    def test_reasoning_accumulates(self):
        turn_trace.add_reasoning("Thinking")
        turn_trace.add_reasoning(" more.")
        self.assertEqual(turn_trace.snapshot()["reasoning"], "Thinking more.")

    def test_empty_reasoning_delta_is_a_no_op(self):
        turn_trace.add_reasoning("")
        self.assertEqual(turn_trace.snapshot()["reasoning"], "")

    def test_tool_call_and_result_are_recorded_in_order(self):
        turn_trace.add_tool_call("get_weather", {"location": "台北"})
        turn_trace.add_tool_result("get_weather", {"forecast": "sunny"})
        steps = turn_trace.snapshot()["steps"]
        self.assertEqual(steps, [
            {"type": "tool_call", "name": "get_weather", "arguments": {"location": "台北"}},
            {"type": "tool_result", "name": "get_weather", "result": {"forecast": "sunny"}},
        ])

    def test_usage_is_recorded(self):
        turn_trace.set_usage(120, 45)
        self.assertEqual(turn_trace.snapshot()["usage"], {"input_tokens": 120, "output_tokens": 45})

    def test_end_marks_done_without_clearing_content(self):
        turn_trace.add_reasoning("x")
        turn_trace.end()
        snap = turn_trace.snapshot()
        self.assertTrue(snap["done"])
        self.assertEqual(snap["reasoning"], "x")

    def test_snapshot_is_a_copy_not_a_live_reference(self):
        snap = turn_trace.snapshot()
        snap["reasoning"] = "mutated"
        self.assertEqual(turn_trace.snapshot()["reasoning"], "")

    def test_snapshots_steps_list_does_not_alias_internal_state(self):
        turn_trace.add_tool_call("get_weather", {})
        snap = turn_trace.snapshot()
        snap["steps"].append({"type": "tool_call", "name": "injected", "arguments": {}})
        self.assertEqual(len(turn_trace.snapshot()["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
