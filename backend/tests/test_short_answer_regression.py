"""A short Chinese answer is an answer.

Bonsai 8B answers "台灣的首都是哪裡？" with a bare "台北" on roughly half of runs.
A `len(text) > 2` filter on candidate assistant messages dropped every one of
those, so the turn fell through to the no-answer path and the user was told
抱歉，我找不到相關的答案 — an apology delivered in place of the correct answer.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.qwen_harness import NO_ANSWER_ZH, _answer_or_fallback, _has_word


class TestHasWord(unittest.TestCase):
    def test_two_character_chinese_answers_are_content(self):
        for answer in ("台北", "是的", "沒有", "五", "對"):
            self.assertTrue(_has_word(answer), f"{answer!r} is a complete answer")

    def test_punctuation_and_list_marks_are_not_content(self):
        for junk in ("", "  ", ".", "。", "1.", "2)", "- ", "***", "[]", "1. 2. 3."):
            self.assertFalse(_has_word(junk), f"{junk!r} is not an answer")

    def test_latin_needs_a_real_word_not_a_stray_letter(self):
        self.assertFalse(_has_word("a."))
        self.assertFalse(_has_word("1a"))
        self.assertTrue(_has_word("Taipei"))
        self.assertTrue(_has_word("yes"))


class TestAnswerOrFallback(unittest.TestCase):
    def test_short_answer_survives_the_filter(self):
        for answer in ("台北", "台北。", "是的"):
            self.assertEqual(_answer_or_fallback(answer), answer,
                             "a short correct answer must not become an apology")

    def test_contentless_residue_still_falls_back(self):
        for junk in ("", "   ", "1.\n2.\n3.", "- \n- "):
            self.assertEqual(_answer_or_fallback(junk), NO_ANSWER_ZH)


if __name__ == "__main__":
    unittest.main()
