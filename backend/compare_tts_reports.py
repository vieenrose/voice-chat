#!/usr/bin/env python3
"""Compare two test_tts_asr_roundtrip.py reports item by item.

Why a separate tool: the interesting question is never 'what is the CER' but 'which
engine mis-says WHICH sentence'. Aggregate means hide exactly the thing a TTS migration
has to be judged on (e.g. one engine is perfect except for mixed-script names).

    python3 backend/compare_tts_reports.py /tmp/rt_audio8.json /tmp/rt_qwen3.json
"""
import json
import os
import statistics
import sys
from collections import defaultdict


def load(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    s = d["summary"]
    items = {}
    rep_totals = defaultdict(list)          # repeat index -> CERs, for the noise band
    for row in d["rows"]:
        if "error" in row and "variants" not in row:
            continue
        for _label, reps in (row.get("variants") or {}).items():
            reps = [r for r in reps if "error" not in r]
            if not reps:
                continue
            for r in reps:
                rep_totals[r.get("rep", 0)].append(r["cer_stream_asr"])
            items[(row["category"], row["text"])] = {
                "cer": min(r["cer_stream_asr"] for r in reps),
                "cers": [r["cer_stream_asr"] for r in reps],
                "hyp": min(reps, key=lambda r: r["cer_stream_asr"])["hyp_stream"],
                "clip": max(r["audio"]["clipped_ratio"] for r in reps),
                "secs": max(r["audio"]["seconds"] for r in reps),
            }
    return s, items, rep_totals


def noise_band(rep_totals):
    """Spread of the whole-corpus mean across repeat indices: the engine+ASR noise floor.

    Without this a 5 pp win on a 6-sample category is unfalsifiable — the TTS samples at
    temperature, so re-measuring the SAME binary moves category means by that much. The
    median of the per-repeat means is the number to compare: one derailment (a TTS that
    repeats a clause scores >200% CER on one item) moves the mean by several points alone.
    """
    means = [sum(v) / len(v) for v in rep_totals.values() if v]
    if len(means) < 2:
        return None
    med = statistics.median(means) if len(means) % 2 else statistics.mean(sorted(means)[len(means) // 2 - 1:len(means) // 2 + 1])
    return {"per_rep_means": [round(m, 4) for m in means],
            "median_rep_mean": round(med, 4),
            "spread_pp": round((max(means) - min(means)) * 100, 1)}


def main(paths):
    if len(paths) < 2:
        raise SystemExit(__doc__)
    loaded = [(os.path.basename(p), *load(p)) for p in paths]
    names = [n for n, _, _, _ in loaded]
    keys = sorted(set.intersection(*[set(items.keys()) for _, _, items, _ in loaded]))
    print(f"engines: {'  vs  '.join(n + ' (' + s['tts_backend'] + ')' for n, s, _, _ in loaded)}")
    for n, _, _, reps in loaded:
        band = noise_band(reps)
        if band:
            print(f"  noise band {n}: corpus mean per repeat {band['per_rep_means']} "
                  f"(median {band['median_rep_mean']:.1%}, spread {band['spread_pp']} pp) "
                  f"— smaller deltas are ties")
    print(f"comparable items: {len(keys)}\n")
    hdr = "  ".join(f"{n[:14]:>14}" for n in names)
    print(f"{'category':12} {'ref text':38} {hdr}   verdict")
    print("-" * (60 + 18 * len(names)))
    wins = defaultdict(int)
    cat_diffs = defaultdict(list)
    for key in keys:
        cat, text = key
        cells, cers = [], []
        for _name, _, items, _reps in loaded:
            it = items[key]
            cers.append(it["cer"])
            cells.append(f"{it['cer']:14.1%}")
        best = min(cers)
        winners = [names[i] for i, c in enumerate(cers) if c <= best + 1e-9]
        verdict = "tie" if len(winners) > 1 else f"{winners[0]} better"
        for w in ([names[cers.index(best)]] if len(winners) == 1 else []):
            wins[w] += 1
        cat_diffs[cat].append(cers)
        print(f"{cat:12} {text[:38]:38} {'  '.join(cells)}   {verdict}")
        if max(cers) - min(cers) > 0.08:
            for name, _, items, _reps in loaded:
                print(f"{'':12} {name[:14]}: {items[key]['hyp'][:100]}")
    print("-" * (60 + 18 * len(names)))
    print("per-category mean CER")
    for cat in sorted(cat_diffs):
        cols = [sum(cs[i] for cs in cat_diffs[cat]) / len(cat_diffs[cat]) for i in range(len(names))]
        print(f"  {cat:12} " + "  ".join(f"{n[:14]:>14} {c:6.1%}" for n, c in zip(names, cols, strict=True)))
    print("\nhead-to-head wins:", dict(wins))
    overall = []
    for _name, _summ, items, _reps in loaded:
        vals = [items[k]["cer"] for k in keys]
        overall.append(sum(vals) / max(1, len(vals)))
    print("overall mean CER: " + "  ".join(f"{n[:14]}={o:.1%}" for n, o in zip(names, overall, strict=True)))
    best = names[overall.index(min(overall))]
    print(f"\n=> lower overall CER: {best} (decide on the per-category rows, not this line)")


if __name__ == "__main__":
    main(sys.argv[1:])
