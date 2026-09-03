"""TTS text normalisation: pronunciation only, never a change of words."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s2s.tts_text import normalize, spell_extensions, to_simplified  # noqa: E402


class TestExtensions(unittest.TestCase):
    def test_a_looked_up_extension_is_spelled_out(self):
        """1102 alone is read 一千一百零二, which is not how an extension is given."""
        self.assertEqual(spell_extensions("分機是 1102。", {"1102"}), "分機是 一一〇二。")

    def test_a_year_is_left_alone(self):
        """Grounded in the directory, so 2026 is not an extension and stays a year."""
        self.assertEqual(spell_extensions("今年是 2026 年。", {"1102"}), "今年是 2026 年。")

    def test_an_extension_never_returned_is_left_alone(self):
        self.assertEqual(spell_extensions("代號 9999", {"1102"}), "代號 9999")

    def test_nothing_looked_up_changes_nothing(self):
        self.assertEqual(spell_extensions("分機是 1102。", set()), "分機是 1102。")

    def test_longer_digit_runs_are_not_touched(self):
        self.assertEqual(spell_extensions("11021 和 011102", {"1102"}), "11021 和 011102")


class TestGlyphs(unittest.TestCase):
    def test_traditional_becomes_simplified(self):
        self.assertEqual(to_simplified("記得帶把傘喔"), "记得带把伞喔")

    def test_conversion_is_character_level_not_vocabulary(self):
        """軟體 must not become the mainland word 软件: that changes what is SAID."""
        self.assertEqual(to_simplified("軟體"), "软体")
        self.assertEqual(to_simplified("程式"), "程式")


class TestTogether(unittest.TestCase):
    def test_both_rules_apply(self):
        self.assertEqual(normalize("她的分機是 1102。", {"1102"}), "她的分机是 一一〇二。")


if __name__ == "__main__":
    unittest.main()
