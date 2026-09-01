"""The streamed answer and the final answer must not both be spoken.

generate_chat_with_tools() emits the harness's deltas live, then reconciles them
against the authoritative `final_text` and speaks only what the stream has not
already said. The reconciliation is a common-prefix comparison, and it was
comparing raw streamed text against a `final_text` that had already been
whitespace-collapsed. Answers routinely open with a newline, so the prefix broke
at character 0, the "tail" became the entire answer, and every streamed reply was
emitted twice — once with its line breaks, once flattened.

That duplicate appeared on every model and quantization tested and was read as a
small-model repetition artifact for a while. It was arithmetic.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def replay_tail(streamed: str, final_text: str) -> str:
    """The reconciliation in llm/ling_streaming.py:generate_chat_with_tools, isolated.

    Mirrors the production logic: normalize the streamed side the same way
    `final_text` was normalized, then return only the un-said remainder.
    """
    streamed_cmp = " ".join(streamed.split())
    common = 0
    for a, b in zip(streamed_cmp, final_text, strict=False):
        if a != b:
            break
        common += 1
    return final_text[common:] if len(final_text) > common else ""


def normalize(text: str) -> str:
    """What generate_chat_with_tools does to final_text before the comparison."""
    return " ".join(text.split())


class TestNothingIsSaidTwice(unittest.TestCase):

    def test_an_answer_opening_with_a_newline_is_not_replayed(self):
        """The exact shape that broke: leading \\n\\n makes a raw comparison fail at 0."""
        streamed = "\n\n根據工具搜尋結果，目前科技界有不少新動態值得關注。\n\n首先，晶片價格調漲。"
        self.assertEqual(replay_tail(streamed, normalize(streamed)), "")

    def test_internal_newlines_alone_are_enough_to_break_a_raw_comparison(self):
        streamed = "台北今天天氣狀況：\n\n最高溫 33°C\n最低溫 26°C"
        self.assertEqual(replay_tail(streamed, normalize(streamed)), "")

    def test_a_fully_streamed_plain_answer_replays_nothing(self):
        streamed = "法國現任總統是艾曼紐·馬克宏。"
        self.assertEqual(replay_tail(streamed, normalize(streamed)), "")

    def test_text_the_stream_never_sent_is_still_spoken(self):
        """The mechanism has to keep working: a final answer that genuinely continues
        past what was streamed must have its remainder spoken, or tool turns go mute."""
        streamed = "台北今天天氣狀況：\n"
        final = normalize("台北今天天氣狀況：\n最高溫 33°C。")
        # The newline the stream sent became a space in `final`, so the separator falls
        # on the un-said side of the split. Harmless — it is whitespace either way.
        self.assertEqual(replay_tail(streamed, final).strip(), "最高溫 33°C。")

    def test_a_final_answer_that_diverges_is_spoken_from_the_divergence(self):
        """Withdrawn/repaired answers differ mid-string; everything from the first
        difference onwards is new and must be said."""
        streamed = "法國總統是誰？我來查一下。"
        final = "法國總統是艾曼紐·馬克宏。"
        tail = replay_tail(streamed, final)
        self.assertTrue(tail.startswith("艾曼紐"), tail)

    def test_no_duplication_for_a_range_of_whitespace_shapes(self):
        for streamed in [
            "答案。",
            "\n答案。",
            "答案。\n\n第二段。",
            "  答案。  \n  第二段。  ",
            "1. 第一\n2. 第二\n3. 第三",
        ]:
            with self.subTest(streamed=streamed):
                self.assertEqual(replay_tail(streamed, normalize(streamed)), "")

    def test_the_production_helper_matches_this_logic(self):
        """Guard against the source drifting away from what this file asserts."""
        path = os.path.join(os.path.dirname(__file__), "..", "llm", "ling_streaming.py")
        src = open(path, encoding="utf-8").read()
        self.assertIn('streamed_cmp = " ".join(streamed.split())', src,
                      "the streamed side must be normalized before the prefix comparison")
        self.assertIn("for a, b in zip(streamed_cmp, final_text", src,
                      "the comparison must use the normalized streamed text")


class TestConcatenatedDuplicateIsDetectable(unittest.TestCase):
    """What the bug looked like from the outside, kept so the symptom is recognizable."""

    def test_the_old_behaviour_produced_the_answer_twice(self):
        streamed = "\n\n台北今天多雲。\n最高溫 33°C。"
        final = normalize(streamed)
        # the old code compared raw streamed text against the collapsed final text
        common = 0
        for a, b in zip(streamed, final, strict=False):
            if a != b:
                break
            common += 1
        old_tail = final[common:]
        self.assertEqual(common, 0, "a leading newline broke the comparison immediately")
        self.assertEqual(old_tail, final, "so the whole answer was re-emitted")
        # and the resulting transcript repeated its sentences
        combined = streamed + old_tail
        sentences = [s for s in re.split(r"[。！？\n]", combined) if s.strip()]
        self.assertNotEqual(len(sentences), len(set(s.strip() for s in sentences)))


if __name__ == "__main__":
    unittest.main()
