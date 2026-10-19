"""Checks the Python PrimeTTS frontend against the iOS golden file (FrontendCheck/golden.json).

usage: python check_frontend.py golden.json <tts resources dir>
Compares: number spelling, text normalization, g2pw query features (ids, position, char, mask),
and the bopomofo per character when the g2pw scorer is replaced by the reference reading.
"""
import json
import os
import sys

from prime_frontend import G2PEn, G2PW, PrimeFrontend, cardinal_en, cardinal_zh, normalize, ordinal_en

golden = json.load(open(sys.argv[1], encoding="utf-8"))
res_dir = sys.argv[2]
failures = 0


def fail(msg):
    global failures
    failures += 1
    print("FAIL:", msg)


for k, v in golden["an2cn"].items():
    if cardinal_zh(int(k)) != v:
        fail("an2cn %s: got %s want %s" % (k, cardinal_zh(int(k)), v))
for k, v in golden["en"]["card"].items():
    if cardinal_en(int(k)) != v:
        fail("card %s: got %s want %s" % (k, cardinal_en(int(k)), v))
for k, v in golden["en"]["ord"].items():
    if ordinal_en(int(k)) != v:
        fail("ord %s: got %s want %s" % (k, ordinal_en(int(k)), v))

resources = json.load(open(os.path.join(res_dir, "g2pw.json"), encoding="utf-8"))
vocab = open(os.path.join(res_dir, "vocab.txt"), encoding="utf-8").read()
g2pw = G2PW(resources, vocab)
symbols = json.load(open(os.path.join(res_dir, "symbol_table.json"), encoding="utf-8"))["symbols"]
g2pen = G2PEn(open(os.path.join(res_dir, "cmudict.txt"), encoding="utf-8").read(),
              os.path.join(res_dir, "g2pen.bin"))

for c in golden["cases"]:
    text = c["text"]
    norm = normalize(text)
    if norm != c["norm"]:
        fail("norm [%s]\n   got  %s\n   want %s" % (text, norm, c["norm"]))

    chars = list(norm)
    want_q = c["queries"]
    got_q = []
    for i, ch in enumerate(chars):
        if ch in resources["char2ph"]:
            q = g2pw._make_query(chars, i)
            if q is not None:
                got_q.append(q)
    if len(got_q) != len(want_q):
        fail("query count [%s]: %d vs %d" % (text, len(got_q), len(want_q)))
    for g, w in zip(got_q, want_q):
        if (g["ids"] != w["input_ids"] or g["position"] != w["position_id"] or g["char_id"] != w["char_id"]
                or sorted(g["allowed"]) != sorted(w["mask_ids"])):
            fail("query %d [%s] ids %s vs %s" % (g["index"], text, g["ids"], w["input_ids"]))

    # reference reading replaces the scorer: checks the surrounding logic, not the ONNX model
    want_bopo = c["bopomofo"]
    cursor = {"row": 0}
    order = [q["index"] for q in got_q]

    def stub(ids, positions, char_ids, masks):
        rows = []
        for _ in ids:
            idx = order[cursor["row"]]
            cursor["row"] += 1
            want = want_bopo[idx]
            row = [0.0] * len(resources["labels"])
            if want in resources["labels"]:
                row[resources["labels"].index(want)] = 1.0
            rows.append(row)
        return rows

    frontend = PrimeFrontend(symbols, g2pw, g2pen, stub)
    got = g2pw.convert(norm, stub)
    if got != want_bopo:
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(got, want_bopo)) if a != b]
        fail("bopomofo [%s] first diffs %s" % (text, diffs[:4]))
    cursor["row"] = 0
    out = frontend.convert(text)
    if "phone_ids" in c and out["phone_ids"] != c["phone_ids"]:
        fail("phone ids [%s]\n   got  %s\n   want %s" % (text, out["phone_ids"][:20], c["phone_ids"][:20]))

print("cases: %d, failures: %d" % (len(golden["cases"]), failures))
sys.exit(1 if failures else 0)
