"""Tests for the TTS text front-end (`tts/spoken_text.py`).

The rule these tests exist to enforce: we change how things are *written*, never what is
*said*. Every assertion below is about a written-language artifact (markdown, a unit
suffix, a percent sign), and the "content is preserved" test is the guard rail against a
rule quietly dropping a word.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts.spoken_text import is_cjk, normalize, speakable  # noqa: E402


class TestMarkdownNeverGetsSpoken(unittest.TestCase):
    def test_bold_and_code_are_stripped_but_words_survive(self):
        out, applied = normalize("重點：**颱風路徑北移**、`停班停課` 名單已公布。")
        self.assertNotIn("*", out)
        self.assertNotIn("`", out)
        for word in ("颱風路徑北移", "停班停課", "名單已公布"):
            self.assertIn(word, out, f"rule deleted content: {out}")
        self.assertIn("md_inline", applied)

    def test_underscore_emphasis_and_headings_and_bullets(self):
        self.assertEqual(speakable("_interesting_"), "interesting")
        self.assertEqual(speakable("## 重點提示"), "重點提示")
        self.assertEqual(speakable("- 第一项\n- 第二项"), "第一项 第二项")
        self.assertEqual(speakable("1. 先做這個\n2. 再做那個"), "先做這個 再做那個")

    def test_link_text_survives_and_url_dropped(self):
        out, applied = normalize("細節見 [中央氣象署](https://www.cwb.gov.tw/V8/) 網站。")
        self.assertIn("中央氣象署", out)
        self.assertNotIn("cwb.gov.tw", out)
        self.assertIn("dropped_link", applied)

    def test_table_rows_keep_their_numbers(self):
        out, _ = normalize("| 台北市 | 34度 |")
        self.assertNotIn("|", out)
        self.assertIn("台北市", out, "a rule that deletes a cell deletes spoken content")
        self.assertIn("34", out)


class TestWrittenSymbolsBecomeSpokenWords(unittest.TestCase):
    def test_celsius_percent_in_chinese(self):
        out, _ = normalize("最高溫度 34°C，最低溫 26°C，降雨機率 68%。")
        self.assertNotIn("°C", out)
        self.assertNotIn("%", out)
        self.assertIn("34度", out)
        self.assertIn("百分之68", out)

    def test_signed_and_ranged_percent(self):
        self.assertIn("增长百分之12", speakable("營收 +12%"))
        ranged = speakable("降雨 20-30%")
        self.assertNotIn("%", ranged, "a bare percent sign must not survive in a zh utterance")
        self.assertIn("百分之20到百分之30", ranged)
        self.assertIn("百分之5", speakable("百分之5%"))          # already spelled: sign is noise

    def test_fraction_reads_denominator_first(self):
        self.assertEqual(speakable("3/4 的人同意"), "4分之3 的人同意")

    def test_english_units(self):
        out, _ = normalize("Revenue rose 12%, and it is 24°C outside.")
        self.assertIn("12 percent", out)
        self.assertIn("24 degrees Celsius", out)
        self.assertNotIn("%", out)

    def test_dollar_phrase_order_english(self):
        self.assertIn("5.2 million dollars", speakable("It cost $5.2 million."))
        self.assertIn("5.2美元", speakable("造價 $5.2 million 元"))

    def test_emoji_dropped(self):
        out, applied = normalize("好 ✅ 呀 🙂")
        self.assertNotIn("✅", out)
        self.assertIn("emoji", applied)


class TestNothingIsInventedOrLost(unittest.TestCase):
    def test_plain_speech_is_unchanged(self):
        for text in ["好的。", "今天天氣不錯，我們下午去河濱公園騎腳踏車。",
                     "The forecast calls for scattered showers this afternoon."]:
            self.assertEqual(speakable(text), text, "clean text must pass through untouched")

    def test_no_rule_contains_a_benchmark_sentence(self):
        """Guard rail against overfitting: rules are patterns, never corpus strings.

        Checks the rule tables themselves (not the docstring, which quotes measured
        examples on purpose) so the check cannot be satisfied by moving a literal.
        """
        import tts.spoken_text as st

        patterns = [p.pattern for tables in (st._ZH_RULES, st._EN_RULES, st._MD_BOLD, st._MD_LINE)
                    for p, _ in tables]
        patterns += [st._MD_CODE[0].pattern, st._MD_LEFTOVER[0].pattern, st._TABLE_OUTER[0].pattern,
                     st._URL.pattern, st._EMOJI.pattern]
        joined = "\n".join(patterns)
        for phrase in ("颱風", "馬克宏", "台積電", "Macron", "quantum", "rainfall", "下雨", "台北"):
            self.assertNotIn(phrase, joined, f"{phrase!r} is corpus text, not a rule")

    def test_empty_and_whitespace(self):
        self.assertEqual(normalize("   \n "), ("", ["empty"]))
        self.assertEqual(speakable(""), "")


class TestScriptDetection(unittest.TestCase):
    def test_is_cjk(self):
        self.assertTrue(is_cjk("台北"))
        self.assertFalse(is_cjk("Taipei"))
        self.assertTrue(is_cjk("IBM 的 paper"), "mixed script must take the CJK rule set")


if __name__ == "__main__":
    unittest.main(verbosity=2)
