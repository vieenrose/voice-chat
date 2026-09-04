"""Is the answer RIGHT -- across the shapes a real user actually asks in.

    python -m s2s.checks.accuracy
    python -m s2s.checks.accuracy --model deepseek-v4-flash    # judge a candidate
    python -m s2s.checks.accuracy --cases followup,negation

Complements the other two accuracy harnesses rather than repeating them:
`s2s.checks.exhaustive` proves each tool ROUTES and that its result reaches the
answer, over the live pipeline; `s2s.checks.extension_eval` grades the attendant
flow at scale. Neither covers the shapes below, which is where a fast-but-sloppy
model shows itself -- a referential follow-up, a question asked in English, a
name the directory does not contain, an instruction the model may quietly
ignore.

Each case says what the answer MUST contain and what it must NOT. A "must not"
is the half that catches fabrication: a model that invents an extension for a
name absent from tools/directory.csv passes any check that only asks whether it
sounded helpful. Judged on the harness's returned text, so a case fails on the
answer the user would hear, not on whether some tool fired.

Runs the harness in-process (like s2s.checks.model_latency, unlike
`exhaustive`), so it needs no free pipeline slot and leaves the running
server's configured model alone -- pass --model to grade a different one.
Needs an OpenCode Go key; see s2s.checks.model_latency for where it is read.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Grade the configuration that actually ships. s2s/deploy/pipeline-cloud.sh pins
# LLM_AGENT_TEMP=0.2, while agent/qwen_harness.py's own default is 0.7 -- and the
# difference is not cosmetic: at 0.7 this suite scored 6/11 against 9/11 at 0.2 on
# the same model, because the extra temperature made the model stop after its
# pre-tool preamble ("好的，我查一下。") instead of going on to call the tool.
# Grading 0.7 would have reported a model nobody runs. Set the env var explicitly
# to override.
os.environ.setdefault("LLM_AGENT_TEMP", "0.2")

from s2s.checks.model_latency import find_key, one_call  # noqa: E402


def zh_num(n: int) -> str:
    """Chinese numeral for 0-59 -- these answers spell numbers out as often not."""
    d = "零一二三四五六七八九"
    if n < 10:
        return d[n]
    if n < 20:
        return "十" + (d[n % 10] if n % 10 else "")
    return d[n // 10] + "十" + (d[n % 10] if n % 10 else "")


def either(n: int) -> str:
    return f"(?:{n}|{zh_num(n)})"


def digits_or_zh(s: str) -> str:
    """Regex matching `s` written with Arabic digits or Chinese numerals."""
    return "".join(either(int(c)) if c.isdigit() else re.escape(c) for c in s)


def now_tpe() -> datetime:
    # The tool reports VOICE_TZ, not the host clock -- this box runs UTC, so
    # judging against localtime would fail a correct answer by 8 hours.
    return datetime.now(ZoneInfo(os.getenv("VOICE_TZ", "Asia/Taipei")))


# (label, history, prompt, must_match, must_not_match, note)
# `history` is real prior turns, so a referential prompt has something to refer to.
def cases() -> list[tuple]:
    t = now_tpe()
    hour = f"(?:{either(t.hour)}|{either((t.hour % 12) or 12)})\\s*[:點時]"
    return [
        ("capital", [], "台灣的首都是哪裡？請簡短回答。",
         "台北", None, "plain knowledge, no tool"),
        ("clock", [], "現在幾點？",
         hour, None, "must be the tool's hour, not the model's guess"),
        ("weather", [], "今天台北天氣如何？",
         r"\d+\s*(?:°C|度)|[一二三四五六七八九十]+\s*度|氣溫|降雨|濕度|溼度|雨|晴|雲",
         None, "must carry the tool's reading"),
        # The referential case the README calls out: history is what makes
        # 「那明天呢？」 answerable at all.
        ("followup",
         [{"role": "user", "content": "今天台北天氣如何？"},
          {"role": "assistant", "content": "今天台北多雲，氣溫二十八度。"}],
         "那明天呢？",
         r"明天|明日|氣溫|度|雨|晴|雲",
         r"你指的是|不清楚你的意思|請問你想問|沒有提到",
         "must resolve the reference, not ask what was meant"),
        # zh-TW whatever the question's language: the README's own promise.
        ("english_in", [], "What time is it in Taipei right now?",
         hour, r"[A-Za-z]{4,}\s+[A-Za-z]{4,}\s+[A-Za-z]{4,}",
         "answer in zh-TW even when asked in English"),
        ("contact_unique", [], "幫我轉接張志強",
         digits_or_zh("1130"), None, "the one 張志強 in the directory is ext 1130"),
        ("contact_ambiguous", [], "幫我轉接陳怡君",
         r"研發|行銷|客服|哪一位|哪個部門|三位|3\s*位",
         None, "three people share this name -- must disambiguate, not pick one"),
        # The fabrication case. No such person exists, so an extension number in
        # the answer is an invented one.
        ("contact_absent", [], "幫我轉接歐陽鐵牛",
         r"找不到|查不到|沒找到|沒有|查無|不在|無此|無法找到",
         r"分機\s*(?:\d{3,4}|[一二三四五六七八九十]{2,})",
         "must not invent an extension for a name that is not there"),
        ("negation", [], "我不要聽天氣，只要告訴我現在幾點。",
         hour, r"氣溫|降雨|濕度|溼度|多雲|晴朗",
         "must honour what the user did NOT ask for"),
        ("math", [], "三加四再乘以五等於多少？請只給答案。",
         f"(?:35|{zh_num(35)})", None, "(3+4)*5 = 35"),
        # The verbosity case: muse-spark-1.2 is terse enough that s2s.checks
        # .bargein's premise (a reply long enough to interrupt) can stop holding.
        ("verbose", [], "請詳細說明台灣半導體產業的歷史、現況與挑戰，分成至少五個段落。",
         r"(?s).{400,}", None, "an explicit 'at least five paragraphs' must be honoured"),
    ]


async def run(model: str | None, key: str, want: list[str], timeout: float) -> list[dict]:
    out = []
    for label, history, prompt, must, must_not, note in cases():
        if want and label not in want:
            continue
        r = await one_call(model or os.getenv("LLM_MODEL_ID", ""), prompt, key, timeout,
                           history=history)
        text = r["answer"]
        hit = bool(re.search(must, text)) if must else True
        bad = bool(re.search(must_not, text)) if must_not else False
        ok = r["ok"] and hit and not bad
        why = "" if ok else ("no answer" if not r["ok"]
                             else "said what it must not" if bad else "missing what it must")
        out.append({"label": label, "ok": ok, "why": why, "note": note,
                    "ttft": r["ttft"], "total": r["total"], "text": text,
                    "tools": r["tools"], "len": len(text)})
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {label:18} {r['total']:5.1f}s {len(text):4d}ch "
              f"{','.join(t for t in r['tools'] if t) or '-':22} {text[:60]!r}")
        if not ok:
            print(f"       -> {why}: {note}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="model id to grade (default: the live server's)")
    ap.add_argument("--cases", default="", help="comma-separated subset of case labels")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    key, src = find_key()
    if not key:
        print("no OpenCode Go key found -- see s2s.checks.model_latency's docstring")
        return 2
    model = args.model
    if not model:
        import json
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8765/v1/llm-config", timeout=10) as r:
            model = json.loads(r.read())["model"]
    want = [c.strip() for c in args.cases.split(",") if c.strip()]
    print(f"grading {model}  (key from {src})\n")

    rows = asyncio.run(run(model, key, want, args.timeout))
    bad = [r for r in rows if not r["ok"]]
    print("\n" + "=" * 78)
    print(f"  {len(rows) - len(bad)}/{len(rows)} correct   model={model}")
    for r in bad:
        print(f"    FAIL {r['label']:18} {r['why']} -- {r['note']}")
        print(f"         said: {r['text'][:150]!r}")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
