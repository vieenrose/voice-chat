"""A small zh-TW staff directory, and the lookup behind `search_contacts`.

Built for the attendant demo, so the data is shaped to make the interesting case
happen rather than to be pretty: **names collide on purpose**. 陳怡君 appears in
three departments, 林建宏 in two, and 張志強/張志祥 differ only by a syllable that
is easy to mishear. A directory where every name is unique would never exercise
the disambiguation turn, which is the part worth demonstrating.

Matching is phonetic as well as literal, because the query arrives from speech:
the model hears 「黃小華」 and may write 「王小華」 (huang/wang), or 「陳」 for 「程」.
Exact hits win; otherwise names are compared by pinyin with rapidfuzz, so a
one-syllable slip still finds the person.

Names here are invented for the demo. Extensions are 4-digit and internal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from pypinyin import Style, lazy_pinyin
from rapidfuzz import fuzz

DEPARTMENTS = ("研發部", "行銷部", "業務部", "人資部", "財務部", "客服部", "設計部", "法務部")


@dataclass(frozen=True)
class Contact:
    name: str
    ext: str
    dept: str
    title: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ruff: noqa: E501
_RAW = [
    # -- deliberate collisions: same name, different departments ---------------
    ("陳怡君", "1101", "研發部", "資深工程師"),
    ("陳怡君", "1102", "行銷部", "行銷企劃"),
    ("陳怡君", "1103", "客服部", "客服專員"),
    ("林建宏", "1110", "業務部", "業務經理"),
    ("林建宏", "1111", "研發部", "韌體工程師"),
    ("王美華", "1120", "財務部", "會計主任"),
    ("王美華", "1121", "人資部", "人資專員"),
    # -- near-homophones: one syllable apart, easy to mishear ------------------
    ("張志強", "1130", "研發部", "架構師"),
    ("張志祥", "1131", "設計部", "視覺設計師"),
    ("黃小華", "1140", "人資部", "招募專員"),
    ("王小華", "1141", "業務部", "業務助理"),
    # -- the rest of the directory --------------------------------------------
    ("李淑芬", "1150", "業務部", "業務副理"),
    ("吳宗翰", "1151", "行銷部", "社群經理"),
    ("劉承恩", "1152", "客服部", "客服主管"),
    ("蔡佩珊", "1153", "設計部", "產品設計師"),
    ("楊敏慧", "1154", "研發部", "測試工程師"),
    ("鄭凱文", "1155", "研發部", "後端工程師"),
    ("許雅婷", "1156", "財務部", "財務分析師"),
    ("謝宗霖", "1157", "法務部", "法務專員"),
    ("洪宥廷", "1158", "人資部", "教育訓練"),
    ("周庭萱", "1159", "行銷部", "品牌企劃"),
    ("徐國豪", "1160", "業務部", "區域業務"),
    ("葉佳蓉", "1161", "客服部", "客服專員"),
    ("莊柏翰", "1162", "研發部", "資料工程師"),
    ("呂欣怡", "1163", "設計部", "介面設計師"),
    ("高志豪", "1164", "法務部", "法務主管"),
    ("簡佑良", "1165", "研發部", "維運工程師"),
]

CONTACTS: list[Contact] = [Contact(n, e, d, t) for n, e, d, t in _RAW]


def _pinyin(text: str) -> str:
    """Toneless pinyin, so a tone confusion (黃/王) still compares equal-ish."""
    return " ".join(lazy_pinyin(text, style=Style.NORMAL))


_PINYIN = {c.name: _pinyin(c.name) for c in CONTACTS}

# Below this, a phonetic match is more likely a different person than a mishearing.
PHONETIC_FLOOR = 80


def search(query: str, department: str = "") -> list[dict]:
    """Contacts matching a spoken name, optionally narrowed by department.

    Returns every match: deciding between them is the caller's job, and for the
    attendant that means asking the user rather than guessing.
    """
    q = (query or "").strip()
    if not q:
        return []
    dept = (department or "").strip()

    pool = [c for c in CONTACTS if not dept or dept in c.dept or c.dept in dept]

    exact = [c for c in pool if c.name == q]
    if exact:
        return [c.as_dict() for c in exact]

    # Substring, for a partial name or a given name on its own.
    partial = [c for c in pool if q in c.name]
    if partial:
        return [c.as_dict() for c in partial]

    # Phonetic: the query came from speech, so a wrong character is expected.
    qp = _pinyin(q)
    scored = [(fuzz.ratio(qp, _PINYIN[c.name]), c) for c in pool]
    hits = sorted(((s, c) for s, c in scored if s >= PHONETIC_FLOOR),
                  key=lambda sc: -sc[0])
    return [c.as_dict() for _, c in hits]


def departments_of(matches: list[dict]) -> list[str]:
    """The distinct departments among matches -- the choices to offer the user."""
    seen: list[str] = []
    for m in matches:
        if m["dept"] not in seen:
            seen.append(m["dept"])
    return seen
