"""Text normalisation for the TTS stage.

Everything here is about how a string should be *pronounced*, so it belongs at
the TTS input and nowhere else. Two rules live here today, and both were learned
from the demo getting them wrong:

**Traditional glyphs are converted to Simplified.** Qwen3-TTS is mainland-trained
and mispronounces Traditional-only characters: 「記得帶把傘喔」 came out with the
wrong syllable in 5 of 6 runs (扇, 散, 線, 三, 山), while the identical sentence
written 记得带把伞哦 was correct 6 of 6. The conversion is character-level, so
軟體 becomes 软体 and NOT the mainland word 软件 -- it changes which glyphs are
read, never the Mandarin that is said.

**Years and extensions are read digit by digit.** 2026年 is 二〇二六年 in Mandarin,
never 兩千零二十六年; the 年 that follows is what identifies it, so 共 2026 元 keeps
its cardinal reading.

**Extensions are read digit by digit.** 1102 is otherwise spoken as 一千一百零二,
which is not how anyone gives an extension. This was first tried inside the
`search_contacts` tool, by handing the model 一一〇二 and asking it to keep that
form; the model re-rendered it as 1102 anyway, and the tool's job is to return
data rather than pronunciation. So the rewrite happens here, and it is grounded
in what was actually looked up: a 4-digit run is rewritten only if it matches an
extension the directory returned this session.

That grounding is only as good as the directory. Extensions originally started at
2000, so 2026 was both a year and someone's extension and the rule fired on both
-- 101 extensions fell between 1900 and 2100. The directory now starts at 3000 and
a test pins it, which removes the ambiguity at the source instead of teaching this
module to guess what a number means.

The transcript the client renders is untouched -- only the audio changes.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
# A year is read digit by digit in Mandarin -- 2026年 is 二〇二六年, never
# 兩千零二十六年. Anchored on the 年 that follows, so 共 2026 元 stays a cardinal.
_YEAR = re.compile(r"(?<!\d)(\d{4})(?=\s*年)")
_CN_DIGITS = "〇一二三四五六七八九"


def _digits(run: str) -> str:
    return "".join(_CN_DIGITS[int(c)] for c in run)


def spell_years(text: str) -> str:
    """2026年 -> 二〇二六年. Unlike an extension this needs no lookup: the 年
    that follows is what makes it a year, and nothing else reads that way."""
    return _YEAR.sub(lambda m: _digits(m.group(1)), text)


def known_extensions() -> set[str]:
    """Extensions the directory returned this session, from the tool trace."""
    try:
        from s2s.tool_trace import snapshot

        return {m.get("ext") for e in snapshot()
                for m in ((e.get("result") or {}).get("matches") or []) if m.get("ext")}
    except Exception:
        logger.exception("could not read the tool trace")
        return set()


def spell_extensions(text: str, known: set[str] | None = None) -> str:
    known = known_extensions() if known is None else known
    if not known:
        return text

    def sub(m: re.Match[str]) -> str:
        d = m.group(1)
        return _digits(d) if d in known else d

    return _FOUR.sub(sub, text)


def to_simplified(text: str) -> str:
    try:
        import zhconv
    except ImportError:
        logger.warning("zhconv missing; TTS will read Traditional glyphs directly")
        return text
    return zhconv.convert(text, "zh-hans")


def normalize(text: str, known: set[str] | None = None) -> str:
    """Everything the TTS should hear differently from what the screen shows."""
    return to_simplified(spell_years(spell_extensions(text, known)))
