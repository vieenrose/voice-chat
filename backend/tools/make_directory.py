"""Generate the 500-row staff directory used by the extension-lookup demo.

Deterministic (fixed seed) so the checked-in CSV can be regenerated and diffed.
Names are drawn from real zh-TW surname frequencies -- 陳林黃張李王 alone are about
a third of Taiwan -- because that frequency is what creates natural collisions and
near-homophones, which is exactly what the lookup has to survive at scale.

The hand-written fixtures from the 27-row version are inserted verbatim and first,
so the tests that pin 陳怡君's three departments keep passing.
"""

import csv
import random
from pathlib import Path

SEED = 20260903
N = 500

# Roughly Taiwan's actual distribution: the top handful dominate.
SURNAMES = [
    ("陳", 11), ("林", 8), ("黃", 6), ("張", 5), ("李", 5), ("王", 4), ("吳", 4),
    ("劉", 3), ("蔡", 3), ("楊", 3), ("許", 2), ("鄭", 2), ("謝", 2), ("郭", 2),
    ("洪", 2), ("邱", 2), ("曾", 2), ("廖", 2), ("賴", 2), ("周", 2), ("徐", 2),
    ("蘇", 1), ("葉", 1), ("莊", 1), ("呂", 1), ("江", 1), ("何", 1), ("蕭", 1),
    ("羅", 1), ("高", 1), ("潘", 1), ("簡", 1), ("朱", 1), ("鍾", 1), ("彭", 1),
    ("游", 1), ("詹", 1), ("胡", 1), ("施", 1), ("沈", 1),
]
GIVEN_1 = list("志明美怡淑建雅家宗佳承柏欣孟宥品冠俊心昱勝彥辰")
GIVEN_2 = list("明華玲芬君宏偉婷豪傑倫如安誠廷翰哲文玉真瑜恩慧強祥")
# English given names: Taiwanese office staff commonly go by one, and callers ask
# for "Stella" without knowing the Chinese name.
ENGLISH = [
    "Stella", "Brian", "Eric", "Kevin", "Jason", "Amy", "Cindy", "David", "Emily",
    "Frank", "Grace", "Henry", "Ivy", "Jack", "Karen", "Leo", "Mandy", "Nick",
    "Olivia", "Peter", "Queenie", "Ryan", "Sandy", "Tony", "Vicky", "Wendy",
    "Alan", "Betty", "Carl", "Daisy", "Ethan", "Fiona", "Gary", "Hazel", "Ian",
    "Jenny", "Kelly", "Lucy", "Michael", "Nancy", "Oscar", "Paul", "Rita",
    "Sam", "Tina", "Victor", "Wayne", "Yvonne", "Zoe", "Angela", "Bella",
    "Chris", "Derek", "Elaine", "Felix", "Gina", "Howard", "Irene", "Jerry",
    "Kate", "Lisa", "Marcus", "Nina", "Owen", "Pearl", "Roger", "Sophia",
    "Terry", "Ursula", "Vincent", "Willa", "Xavier", "Yuki", "Zack", "Aaron",
    "Bruce", "Claire", "Dennis", "Eva", "Fred", "Gloria", "Hugo", "Iris",
    "Joyce", "Kenny", "Laura", "Martin", "Nelson", "Ophelia", "Patrick",
    "Rachel", "Simon", "Tracy", "Ulysses", "Vera", "Walter", "Yolanda",
]
# Not everyone goes by an English name, and the ones who do are known by it -- so
# they are handed out WITHOUT replacement. A caller asking for "Stella" expects one
# person; two Stellas would make the name useless as a query.
DEPARTMENTS = ["研發部", "行銷部", "業務部", "人資部", "財務部", "客服部", "設計部", "法務部"]
TITLES = {
    "研發部": ["軟體工程師", "韌體工程師", "測試工程師", "架構師", "資料工程師", "維運工程師"],
    "行銷部": ["行銷企劃", "社群經理", "品牌企劃", "活動企劃"],
    "業務部": ["業務代表", "業務經理", "業務副理", "區域業務"],
    "人資部": ["人資專員", "招募專員", "教育訓練", "薪酬專員"],
    "財務部": ["會計專員", "會計主任", "財務分析師", "出納"],
    "客服部": ["客服專員", "客服主管", "技術支援"],
    "設計部": ["視覺設計師", "產品設計師", "介面設計師"],
    "法務部": ["法務專員", "法務主管", "智財專員"],
}

# Verbatim from the hand-built version: the collisions and near-homophones the
# tests pin. Kept first so their extensions never move.
FIXTURES = [
    ("陳怡君", "1101", "研發部", "資深工程師"),
    ("陳怡君", "1102", "行銷部", "行銷企劃"),
    ("陳怡君", "1103", "客服部", "客服專員"),
    ("林建宏", "1110", "業務部", "業務經理"),
    ("林建宏", "1111", "研發部", "韌體工程師"),
    ("王美華", "1120", "財務部", "會計主任"),
    ("王美華", "1121", "人資部", "人資專員"),
    ("張志強", "1130", "研發部", "架構師"),
    ("張志祥", "1131", "設計部", "視覺設計師"),
    ("黃小華", "1140", "人資部", "招募專員"),
    ("王小華", "1141", "業務部", "業務助理"),
]


def main() -> None:
    rng = random.Random(SEED)
    pool = [s for s, w in SURNAMES for _ in range(w)]
    # Fixtures gain an English name too; the tests pin name/ext/dept only.
    rows = [(n, e, d, t, en) for (n, e, d, t), en in
            zip(FIXTURES, ["Stella", "Cindy", "Ivy", "Brian", "Kevin", "Grace",
                           "Wendy", "Jason", "Sandy", "Amy", "Leo"], strict=True)]
    used_ext = {r[1] for r in rows}
    seen = [(r[0], r[2]) for r in rows]

    english = rng.sample(ENGLISH, len(ENGLISH))
    for r in rows:                       # fixtures already carry theirs
        if r[4] in english:
            english.remove(r[4])
    # Extensions start above the year range on purpose. The TTS normaliser spells
    # out a 4-digit run when it matches an extension this session looked up, and
    # with extensions at 2000+ that rule also caught the year 2026 -- 101 of them
    # fell between 1900 and 2100. Keeping the directory clear of year-like numbers
    # removes the collision at the source instead of adding a rule about years.
    ext = 3000
    while len(rows) < N:
        name = rng.choice(pool) + rng.choice(GIVEN_1) + rng.choice(GIVEN_2)
        dept = rng.choice(DEPARTMENTS)
        # One person per name per department: two 陳怡君 in 行銷部 would be
        # unanswerable by department, which is the only question the bot can ask.
        if (name, dept) in seen:
            continue
        seen.append((name, dept))
        while str(ext) in used_ext:
            ext += 1
        rows.append((name, str(ext), dept, rng.choice(TITLES[dept]),
                     english.pop() if english else ""))
        used_ext.add(str(ext))
        ext += 1

    out = Path(__file__).with_name("directory.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "ext", "dept", "title", "english"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
