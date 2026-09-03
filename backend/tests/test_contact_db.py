"""The directory lookup, and the ambiguity it is supposed to preserve."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.contact_db import CONTACTS, departments_of, search  # noqa: E402


class TestSearch(unittest.TestCase):
    def test_a_colliding_name_returns_every_match(self):
        """The whole point of the attendant flow: do not pick one silently."""
        m = search("陳怡君")
        self.assertEqual(len(m), 3)
        self.assertEqual(departments_of(m), ["研發部", "行銷部", "客服部"])

    def test_department_narrows_to_one(self):
        m = search("陳怡君", "行銷部")
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0]["ext"], "1102")

    def test_a_misheard_name_still_finds_the_person(self):
        """程/陳 is one of the confusions speech input actually produces."""
        m = search("程怡君")
        self.assertTrue(m, "phonetic fallback should match 陳怡君")
        self.assertEqual({c["name"] for c in m}, {"陳怡君"})

    def test_exact_match_beats_a_homophone(self):
        """王小華 and 黃小華 are different people; an exact hit must not blur them."""
        m = search("王小華")
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0]["dept"], "業務部")

    def test_unknown_name_returns_nothing(self):
        self.assertEqual(search("莫須有"), [])

    def test_empty_query_is_not_a_wildcard(self):
        """An argument-less call must not dump the whole directory."""
        self.assertEqual(search(""), [])
        self.assertEqual(search("   "), [])

    def test_extensions_are_unique(self):
        exts = [c.ext for c in CONTACTS]
        self.assertEqual(len(exts), len(set(exts)))

    def test_no_extension_looks_like_a_year(self):
        """The TTS normaliser spells out any 4-digit run that is an extension.

        With extensions starting at 2000, that rule also caught the year 2026 --
        101 of them fell in 1900-2100. The directory keeps clear of that range so
        the ambiguity cannot arise, rather than the normaliser guessing what a
        number means.
        """
        yearish = [c.ext for c in CONTACTS if "1900" <= c.ext <= "2100"]
        self.assertEqual(yearish, [], "extensions must not look like years")


if __name__ == "__main__":
    unittest.main()
