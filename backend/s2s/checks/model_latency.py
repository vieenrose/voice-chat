"""Which OpenCode Go model answers fastest, and what that speed costs.

    python -m s2s.checks.model_latency                     # whole catalogue, 2 reps
    python -m s2s.checks.model_latency --reps 3
    python -m s2s.checks.model_latency --max-input-price 0.40   # the cheap tier only
    python -m s2s.checks.model_latency --models glm-5.3-flash,mimo-v2.5
    python -m s2s.checks.model_latency --prompt "今天台北天氣如何？"

Drives this project's OWN harness (agent.qwen_harness.run_agent_task -> the
real system prompt, the real four tools, the wire format _wire_format picks per
model) once per model, in THIS process. Deliberately not through the pipeline:
`s2s.checks.latency` measures the whole stack but holds the single session slot
and adds TTS, and the README's own numbers say the hosted call dominates a turn
on this branch -- so a per-model comparison wants the hosted call isolated,
and wants to leave the running server's own configured model untouched.

Reported per model:
  ttft   seconds to the first answer delta -- what the pipeline would hand TTS,
         so this, not `total`, is what a listener waits through in silence
  total  seconds until the harness returns the finished answer
  tok    measured input/output tokens for the turn (from the llm_usage event)
  $/turn those measured tokens priced at the model's published Go rate

Prices are hardcoded from https://opencode.ai/docs/go (fetched 2026-09-04) and
are the LAST thing to trust here: the API's /models endpoint carries no pricing
at all, the docs quote ranges for some models (this table takes the top of a
range, so a cost comparison errs against the pricier model rather than for it),
and several catalogue entries are not priced in the docs at all -- those print
"?" and are ranked on latency alone. Go bills $10/month against a metered
allowance at these rates, so a rate difference is real money, not a list price.

Needs an OpenCode Go key, which is NOT the key s2s.checks.exhaustive reads:
~/.openrouter_key holds an OpenRouter key (sk-o...), and OpenCode Go 401s on it.
Looked for at $OPENCODE_GO_KEY_FILE, then ~/.opencode_go_key, then
~/.openrouter_key, or pass --key-file. A key pasted into the UI is unreachable
from here on purpose: it lives only in the server process's environment, never
on disk (see s2s/serve.py's /v1/llm-config), so it cannot be borrowed for a
sweep -- the sweep needs its own copy.

Costs one hosted call per model per rep -- a full sweep is ~50 cheap calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE = os.getenv("S2S_HTTP_BASE", "http://127.0.0.1:8765")
OPENCODE_GO_BASE = "https://opencode.ai/zen/go/v1"
# In preference order. The OpenRouter file is last and is only a courtesy: it is
# a different provider's key, so it 401s here -- see the module docstring.
KEYFILES = [
    os.getenv("OPENCODE_GO_KEY_FILE", ""),
    "~/.opencode_go_key",
    "~/.openrouter_key",
]

# (input, output) US$ per million tokens. See the docstring's caveats: ranges are
# taken at their top, and an absent entry means the docs do not price it.
PRICES: dict[str, tuple[float, float]] = {
    "muse-spark-1.2-contributor": (0.10, 0.20),
    "muse-spark-1.3-contributor": (0.10, 0.20),
    "mimo-v2.5": (0.14, 0.28),
    "glm-5.3-flash": (0.15, 0.50),
    "deepseek-v4-flash": (0.22, 0.66),
    "longcat-2.0": (0.30, 1.20),
    "gpt-5.6-luna": (0.40, 1.80),
    "deepseek-v4-pro": (1.32, 3.96),
    "grok-4.5": (4.00, 12.00),
    "grok-4.6": (4.00, 12.00),
    "kimi-k3": (3.00, 15.00),
}

# The tool-calling shape: two hosted round-trips (decide to call, then answer
# from the result), which is the realistic worst case for a voice turn and the
# shape the README's own 3.97s/8.60s first-audio numbers were measured on.
DEFAULT_PROMPT = "現在幾點？"


def find_key(explicit: str = "") -> tuple[str, Path | None]:
    """(key, where it came from). Empty key means nothing usable was found."""
    for cand in ([explicit] if explicit else KEYFILES):
        if not cand:
            continue
        p = Path(cand).expanduser()
        if p.exists() and (k := p.read_text().strip()):
            return k, p
    return "", None


def catalogue() -> list[str]:
    """The live, already-filtered catalogue the UI's dropdown shows."""
    with urllib.request.urlopen(f"{BASE}/v1/llm-models", timeout=20) as r:
        return list(json.loads(r.read()).get("models") or [])


async def one_call(model: str, prompt: str, key: str, timeout: float,
                    history: list[dict] | None = None) -> dict:
    """One harness turn against `model`, timed. Never raises.

    `history` is real prior {role, content} turns, passed straight through to the
    harness so a referential prompt has something to resolve against -- see
    s2s.checks.accuracy's followup case.
    """
    from agent._shared import set_emit_target
    from agent.qwen_harness import run_agent_task

    os.environ["LLM_API_BASE"] = OPENCODE_GO_BASE
    os.environ["LLM_MODEL_ID"] = model
    os.environ["LLM_API_KEY"] = key

    q: asyncio.Queue = asyncio.Queue()
    set_emit_target(asyncio.get_running_loop(), q)
    out: dict = {"model": model, "ttft": None, "tokens_in": None, "tokens_out": None,
                 "tools": [], "answer": "", "ok": False}
    t0 = time.time()
    task = asyncio.create_task(run_agent_task(prompt, q, history=history))

    # Drain events while the turn runs: the first llm_delta is the TTFT this
    # measures, and llm_usage is the only place the real token counts appear.
    async def drain() -> None:
        while True:
            ev = await q.get()
            kind = ev.get("type")
            if kind == "llm_delta" and out["ttft"] is None and (ev.get("text") or "").strip():
                out["ttft"] = time.time() - t0
            elif kind == "llm_usage":
                out["tokens_in"] = ev.get("input_tokens")
                out["tokens_out"] = ev.get("output_tokens")
            elif kind == "tool_call":
                out["tools"].append(ev.get("name"))

    drainer = asyncio.create_task(drain())
    try:
        answer = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
        answer = ""
        out["error"] = f"no answer within {timeout:.0f}s"
        task.cancel()
    except Exception as e:
        answer = ""
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        await asyncio.sleep(0)      # let the drainer pick up anything still queued
        drainer.cancel()
    out["total"] = time.time() - t0
    out["answer"] = (answer or "").strip()
    # Every harness failure path returns Chinese prose starting 抱歉 rather than
    # raising (agent.qwen_harness._provider_failure_zh / NO_ANSWER_ZH), so a
    # model that cannot serve this project at all still "answers" -- the same
    # convention s2s.checks.exhaustive treats as a refusal.
    out["ok"] = bool(out["answer"]) and not out["answer"].startswith("抱歉")
    return out


def cost(model: str, tin: int | None, tout: int | None) -> float | None:
    p = PRICES.get(model)
    if not p or tin is None or tout is None:
        return None
    return tin / 1e6 * p[0] + tout / 1e6 * p[1]


def report(runs: list[dict]) -> None:
    by_model: dict[str, list[dict]] = {}
    for r in runs:
        by_model.setdefault(r["model"], []).append(r)

    rows = []
    for model, rs in by_model.items():
        good = [r for r in rs if r["ok"] and r["ttft"] is not None]
        if not good:
            rows.append({"model": model, "ttft": None,
                         "why": rs[0].get("error") or (rs[0]["answer"][:38] or "no answer")})
            continue
        tin = statistics.median([r["tokens_in"] for r in good if r["tokens_in"]] or [0])
        tout = statistics.median([r["tokens_out"] for r in good if r["tokens_out"]] or [0])
        rows.append({
            "model": model,
            "ttft": statistics.median([r["ttft"] for r in good]),
            "ttft_min": min(r["ttft"] for r in good),
            "ttft_max": max(r["ttft"] for r in good),
            "total": statistics.median([r["total"] for r in good]),
            "n": len(good),
            "of": len(rs),
            "tin": int(tin), "tout": int(tout),
            "cost": cost(model, int(tin), int(tout)),
            "tools": sorted({t for r in good for t in r["tools"] if t}),
        })

    ok_rows = sorted([r for r in rows if r["ttft"] is not None], key=lambda r: r["ttft"])
    bad_rows = [r for r in rows if r["ttft"] is None]

    print("\n" + "=" * 96)
    print("  PER-MODEL LATENCY (median of reps; ttft = first answer delta, what TTS would speak)")
    print("=" * 96)
    print(f"  {'model':30} {'ttft':>7} {'range':>15} {'total':>7} {'tok in/out':>13} "
          f"{'$/turn':>9}  tool")
    for r in ok_rows:
        c = f"${r['cost']:.5f}" if r["cost"] is not None else "?"
        print(f"  {r['model']:30} {r['ttft']:6.2f}s "
              f"{r['ttft_min']:6.2f}-{r['ttft_max']:5.2f}s {r['total']:6.2f}s "
              f"{r['tin']:6d}/{r['tout']:<6d} {c:>9}  {','.join(r['tools']) or '-'}"
              + ("" if r["n"] == r["of"] else f"   ({r['n']}/{r['of']} ok)"))
    if bad_rows:
        print("\n  did not answer:")
        for r in bad_rows:
            print(f"    {r['model']:30} {r['why']}")
    print("=" * 96)
    if ok_rows:
        priced = [r for r in ok_rows if r["cost"] is not None]
        print(f"  fastest overall      : {ok_rows[0]['model']} at {ok_rows[0]['ttft']:.2f}s")
        if priced:
            best = min(priced, key=lambda r: r["ttft"])
            cheap = min(priced, key=lambda r: r["cost"])
            print(f"  fastest with a price : {best['model']} at {best['ttft']:.2f}s, "
                  f"${best['cost']:.5f}/turn")
            print(f"  cheapest             : {cheap['model']} at ${cheap['cost']:.5f}/turn, "
                  f"{cheap['ttft']:.2f}s")
        print("=" * 96)


async def sweep(models: list[str], prompt: str, reps: int, key: str, timeout: float) -> list[dict]:
    runs = []
    for model in models:
        for i in range(reps):
            r = await one_call(model, prompt, key, timeout)
            runs.append(r)
            ttft = f"{r['ttft']:.2f}s" if r["ttft"] is not None else "  -  "
            status = "ok" if r["ok"] else (r.get("error") or r["answer"][:30] or "no answer")
            print(f"  {model:30} rep {i + 1}/{reps}  ttft={ttft:>7}  "
                  f"total={r['total']:6.2f}s  {status}")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated ids (default: live catalogue)")
    ap.add_argument("--max-input-price", type=float, default=None, metavar="USD",
                     help="only models whose published input rate is <= this, per Mtok. "
                          "Drops the unpriced ones too, since 'reasonable cost' cannot be "
                          "claimed for a rate the docs do not publish.")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds before a turn is a hang")
    ap.add_argument("--key-file", default="", help="OpenCode Go key file (see module docstring)")
    args = ap.parse_args()

    key, src = find_key(args.key_file)
    if not key:
        tried = ", ".join(c for c in ([args.key_file] if args.key_file else KEYFILES) if c)
        print(f"no OpenCode Go key found (tried: {tried})")
        print("A key pasted into the UI cannot be reused here -- it lives only in the "
              "server process's environment, never on disk. Write one to "
              "~/.opencode_go_key or pass --key-file.")
        return 2
    # Fingerprint only, in the same masked form /v1/llm-config publishes: enough
    # to tell two keys apart, never enough to use one.
    print(f"key from {src}: {key[:4]}…{key[-4:]}")
    models = ([m.strip() for m in args.models.split(",") if m.strip()] if args.models
              else catalogue())
    if args.max_input_price is not None:
        models = [m for m in models
                  if m in PRICES and PRICES[m][0] <= args.max_input_price]
    print(f"{len(models)} models x {args.reps} reps  prompt={args.prompt!r}\n")
    runs = asyncio.run(sweep(models, args.prompt, args.reps, key, args.timeout))
    report(runs)
    return 0 if any(r["ok"] for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
