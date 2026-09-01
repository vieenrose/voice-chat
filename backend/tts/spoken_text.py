"""
Spoken-text front-end: turn written answers into something a TTS can say.

Why this exists (measured, not guessed)
---------------------------------------
`test_tts_asr_roundtrip.py` on the engine that shipped with this repo:

    mixed     "IBM 的 quantum 團隊發表了新 paper，_interesting_ 吧？"  CER 300%
    markdown  "重點：**颱風路徑北移**、`停班停課` 名單已公布。"          CER  82%
    numbers   "最高溫度 34°C，最低溫 26°C，降雨機率 68%。"              CER  85%
    plain_zh  "今天天氣不錯，我們下午去河濱公園騎腳踏車。"                CER  ~10%
    pure_en   "The forecast for Tokyo calls for ..."                     CER   0%

The failures are not the kind a bigger model fixes: they are *written-language*
artifacts — markdown emphasis, code spans, unit suffixes, percent signs — reaching an
acoustic model that was trained on read speech. The LLM writes markdown because that is
what chat models do; until now the pipeline sent it straight to the TTS.

Design rules
------------
1. Never change *content*. Nothing here rewrites a word, reorders a clause, or drops
   anything but markup. When a rule must drop spoken information (a URL), it is reported
   in the returned diagnostics rather than done silently.
2. Language-aware, not sentence-aware. Rules key off "does this contain CJK", never off a
   specific string. There is no per-question table in this file — that would be benchmark
   gaming, and it rots the moment the model's phrasings change.
3. One funnel: every caller that hands text to a TTS goes through `normalize()`, so the
   voice path and the text path cannot drift apart.

Rule ORDER matters: specific shapes ("+60%", "20-30%") run before the generic "%" rule,
or the generic rewrite destroys the pattern the specific one needs.
"""
from __future__ import annotations

import re

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_URL = re.compile(r"""https?://\S+|www\.[^\s]+\.(?:com|org|net|tw|io)\S*""", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\uFE0F"
    "\u2B00-\u2BFF\u2190-\u21FF]"
)
_MD_BOLD = [
    (re.compile(r"\*\*\*(.+?)\*\*\*", re.S), r"\1"),
    (re.compile(r"\*\*(.+?)\*\*", re.S), r"\1"),
    (re.compile(r"__(.+?)__", re.S), r"\1"),
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", re.S), r"\1"),
    (re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", re.S), r"\1"),
]
_MD_CODE = (re.compile(r"`{1,3}([^`]*)`{1,3}"), r"\1")
_MD_LINK = [
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
]
_MD_LINE = [
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),
    (re.compile(r"^\s{0,3}[>*+-]\s+", re.M), ""),
    (re.compile(r"^\s*(?:\d+[.)、])\s*", re.M), ""),
    (re.compile(r"^\s*(?:-{3,}|={3,})\s*$", re.M), ""),
]
_MD_LEFTOVER = (re.compile(r"[#*_~]{2,}"), "")
# Strip only the OUTER pipes of a table row; inner ones fall through to the `|` -> "，"
# symbol rule. A rule that matched the whole row passed review until a test proved it would
# have silently deleted the numbers being spoken.
_TABLE_OUTER = (re.compile(r"^[ \t]*\|[ \t]*|[ \t]*\|[ \t]*$", re.M), "")

_ZH_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(\d)\s*(?:°C|℃)"), r"\1度"),
    (re.compile(r"(\d)\s*(?:°F|℉)"), r"\1华氏度"),
    (re.compile(r"(\d)\s*°(?!\s*[FC])"), r"\1度"),
    # already spelled out with 百分之: only the sign is redundant
    (re.compile(r"百分之\s*(\d+(?:\.\d+)?)\s*%"), r"百分之\1"),
    (re.compile(r"\+(\d+(?:\.\d+)?)\s*%"), r"增长百分之\1"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*[-–—~～]\s*(\d+(?:\.\d+)?)\s*%"), r"\1%到\2%"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*%"), r"百分之\1"),
    (re.compile(r"(?<![\da-zA-Z])(\d+(?:\.\d+)?)\s*pp\b"), r"\1个百分点"),
    (re.compile(r"(\d+)\s*[~～–—]\s*(\d+)"), r"\1到\2"),
    # 3/4 -> "4分之3": Chinese reads the denominator first, so the swap IS the rule.
    (re.compile(r"(\d+)\s*/\s*(\d+)"), r"\2分之\1"),
    (re.compile(r"(?<=\d)km/h"), "公里每小时"),
    (re.compile(r"(?<=\d)km\b"), "公里"),
    (re.compile(r"(?<=\d)kg\b"), "公斤"),
    (re.compile(r"(?<=\d)MB\b", re.I), "兆"),
    (re.compile(r"(?<=\d)GB\b", re.I), "吉字节"),
    (re.compile(r"\$\s*([\d,.]+)"), r"\1美元"),
    (re.compile(r"\s*[→➡]\s*"), "到"),
    (re.compile(r"\s*≈\s*"), "大约"),
    (re.compile(r"\s*&\s*"), "和"),
    (re.compile(r"\s*[|┃｜]\s*"), "，"),
    (re.compile(r"[【】《》]"), ""),
]

_EN_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(\d)\s*(?:°C|℃)"), r"\1 degrees Celsius"),
    (re.compile(r"(\d)\s*(?:°F|℉)"), r"\1 degrees Fahrenheit"),
    (re.compile(r"\+(\d+(?:\.\d+)?)\s*%"), r"up \1 percent"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*%"), r"\1 percent"),
    (re.compile(r"(\d+)\s*[~～–—]\s*(\d+)"), r"\1 to \2"),
    # "$5.2 million" must not become "5.2 dollars million" — the qualifier keeps the order.
    (re.compile(r"\$\s*([\d,.]+)\s*(million|billion|trillion)?", re.I),
     lambda m: f"{m.group(1)} {m.group(2) or ''} dollars".replace("  ", " ").strip()),
    (re.compile(r"(?<=\d)km/h"), " kilometers per hour"),
    (re.compile(r"(?<=\d)km\b"), " kilometers"),
    (re.compile(r"(?<=\d)kg\b"), " kilograms"),
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"\s*[→➡≈]\s*"), " to "),
    (re.compile(r"\s*[|┃｜]\s*"), ", "),
]


def is_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def normalize(text: str) -> tuple[str, list[str]]:
    """Return (speakable_text, applied_rule_names). Empty/whitespace input yields ("", ...)."""
    applied: list[str] = []
    if not text or not text.strip():
        return "", ["empty"]
    out = text
    had_link = bool(_URL.search(out) or _EMAIL.search(out))

    def _sub(pat: re.Pattern, rep: str, tag: str, *, count: int = 0) -> None:
        nonlocal out
        new = pat.sub(rep, out, count)
        if new != out:
            out = new
            if tag not in applied:
                applied.append(tag)

    # 1. Markup that must never be voiced.
    for pat, rep in _MD_LINE:
        _sub(pat, rep, "md_line")
    for pat, rep in _MD_BOLD + [_MD_LINK[0], _MD_LINK[1], _MD_CODE]:
        _sub(pat, rep, "md_inline")
    _sub(*_MD_LEFTOVER, tag="md_leftover")
    _sub(*_TABLE_OUTER, tag="md_table")

    # 2. Things a speaker cannot say. A dropped URL is content loss -> reported, including
    #    when the link markup rule above already swallowed it together with its target.
    if had_link and not (_URL.search(out) or _EMAIL.search(out)):
        applied.append("dropped_link")
    _sub(_URL, " ", "dropped_link")
    _sub(_EMAIL, " ", "dropped_link")
    if _EMOJI.search(out):
        _sub(_EMOJI, " ", "emoji")

    # 3. Written-only symbols -> spoken words, per script (not per sentence).
    for pat, rep in (_ZH_RULES if is_cjk(out) else _EN_RULES):
        _sub(pat, rep, f"symbol:{pat.pattern[:18]}")

    # 4. Whitespace and stuttered punctuation.
    _sub(re.compile(r"[ \t\u00a0]+"), " ", "ws")
    _sub(re.compile(r"[，,]{2,}"), "，", "ws")
    _sub(re.compile(r"[。．.]{2,}"), "。", "ws")
    return re.sub(r"\s+", " ", out).strip(" \t\r\n"), applied


def speakable(text: str) -> str:
    """Convenience wrapper for call sites that don't care what was applied."""
    return normalize(text)[0]
