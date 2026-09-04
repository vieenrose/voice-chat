"""End-to-end latency measurement against the LIVE Realtime pipeline.

    python -m s2s.checks.latency                        # default prompts, 3 reps each
    python -m s2s.checks.latency --reps 5
    python -m s2s.checks.latency --labels clock,weather
    python -m s2s.checks.latency --max-first-audio 60

Measures wall-clock time from response.create to the first
response.output_audio.delta (TTFA -- what a listener actually waits through
in silence) and to response.done (total), repeated across the same prompt
shapes s2s.checks.exhaustive already exercises for correctness: one that
needs no tool, and the three tool-calling ones the README's "Measured"
section reports numbers for (clock/weather/search). Reuses one_turn() from
that module rather than re-walking the Realtime protocol a second time.

This branch's own numbers (README "Measured") show first-audio time for a
turn against the hosted LLM swinging roughly 4-9s call to call, and the
README is explicit that the spread is the hosted provider's own load, not
anything measurable or controllable from this side of the API. A fixed
"must beat Xms" target would therefore fail on a slow provider tick through
no fault of this codebase -- the only honest thing to gate on is a turn that
never answers at all (a hang), so PASS/FAIL here is "nothing exceeded a
generous ceiling", and the printed percentiles are for a human to read the
trend, not for the exit code to judge.

Needs the pipeline up on :8765 (see s2s/serve.py) and a valid OpenCode Go key
already POSTed to /v1/llm-config. Close any browser tab first -- one session
slot, and a run of N reps across all four prompts spends 4*N hosted calls.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s2s.checks.exhaustive import TURNS, http, one_turn, wait_for_slot  # noqa: E402

# Two hosted round-trips (initial call + the post-tool-result call a tool turn
# needs) at the single-call timeout this project already enforces, plus slack
# for TTS -- not a performance target, just "did this hang".
DEFAULT_CEILING = 2 * float(os.getenv("LLM_REQUEST_TIMEOUT", "45")) + 10


def _pct(values: list[float], p: int) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


async def run(labels: list[str], reps: int, ceiling: float) -> list[dict]:
    prompts = [(label, prompt) for label, prompt, _needs_fact in TURNS if label in labels]
    runs: list[dict] = []
    for label, prompt in prompts:
        for i in range(reps):
            if not wait_for_slot():
                runs.append({"label": label, "ok": False, "error": "no free pipeline slot"})
                continue
            try:
                r = await one_turn(prompt, timeout=ceiling)
            except Exception as e:
                runs.append({"label": label, "ok": False, "error": f"{type(e).__name__}: {e}"})
                continue
            ok = bool(r["text"].strip()) and r["first_audio"] is not None and r["total"] <= ceiling
            record = {"label": label, "ok": ok, "first_audio": r["first_audio"], "total": r["total"]}
            runs.append(record)
            fa = f"{r['first_audio']:.2f}s" if r["first_audio"] is not None else "never"
            print(f"  {label:8} rep {i + 1}/{reps}: first_audio={fa:>8}  total={r['total']:.2f}s"
                  + ("" if ok else "  FAIL"))
    return runs


def report(runs: list[dict]) -> bool:
    print("\n" + "=" * 66)
    print("  E2E LATENCY (time to first audio, per prompt shape)")
    print("=" * 66)
    by_label: dict[str, list[float]] = {}
    for r in runs:
        if r.get("ok") and r["first_audio"] is not None:
            by_label.setdefault(r["label"], []).append(r["first_audio"])
    for label, values in by_label.items():
        print(f"  {label:8} n={len(values):<3} avg={statistics.mean(values):6.2f}s"
              f"  p50={statistics.median(values):6.2f}s  p95={_pct(values, 95):6.2f}s"
              f"  min={min(values):6.2f}s  max={max(values):6.2f}s")
    all_values = [v for values in by_label.values() for v in values]
    if all_values:
        print(f"  {'overall':8} n={len(all_values):<3} avg={statistics.mean(all_values):6.2f}s"
              f"  p50={statistics.median(all_values):6.2f}s  p95={_pct(all_values, 95):6.2f}s")
    failed = [r for r in runs if not r.get("ok")]
    if failed:
        print("\n  failures:")
        for r in failed:
            detail = r.get("error") or f"total={r.get('total')}s first_audio={r.get('first_audio')}"
            print(f"    [{r['label']}] {detail}")
    print("=" * 66)
    return not failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="repetitions per prompt shape")
    ap.add_argument("--labels", default="chat,clock,weather,search",
                     help="comma-separated subset of TURNS labels to run")
    ap.add_argument("--max-first-audio", type=float, default=DEFAULT_CEILING,
                     help="seconds a turn may take before it counts as a hang")
    args = ap.parse_args()

    st, cfg = http("/v1/llm-config")
    if st == 200:
        print(f"model under test: {cfg.get('model')} @ {cfg.get('provider')}"
              f"  key={'set' if cfg.get('key_set') else 'ABSENT -- no answers, by design'}")
    print(f"reps={args.reps}  ceiling={args.max_first_audio:.0f}s  labels={args.labels}\n")

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    runs = asyncio.run(run(labels, args.reps, args.max_first_audio))
    return 0 if report(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
