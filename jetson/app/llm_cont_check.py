"""Multi-turn reuse check: every turn is answered twice, once as a "+" continuation of the engine's
state and once with a full prefill; greedy decoding must give the same reply, and the continuation
must prefill fewer tokens. Needs the GPU to itself (stop the server first).

  venv/bin/python -u app/llm_cont_check.py ./fwd35_serve
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import Llm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TURNS = ["你好", "我想請教一個問題", "請問台灣的首都是哪裡？", "那台南在台灣的哪裡？", "謝謝你"]


def answer(llm, history, text):
    t0 = time.monotonic()
    first = None
    reply = ""
    for d in llm.stream(history, text):
        if first is None:
            first = time.monotonic() - t0
        reply += d
    return reply, first


def main(binary):
    llm = Llm(binary, str(ROOT / "models" / "Ternary-Bonsai-2-2B-PTQ1_0.gguf"),
              str(ROOT / "models" / "tok" / "tokenizer.json"), max_new=60)
    history, bad = [], 0
    for text in TURNS:
        cont, t_cont = answer(llm, history, text)
        reused = llm.reused
        llm.held = []  # forces a full prefill of the same prompt
        full, t_full = answer(llm, history, text)
        same = cont == full
        bad += not same
        print("%s %-14s reused=%-4d first token: cont %.2fs  full %.2fs" % (
            "SAME" if same else "DIFF", text, reused, t_cont, t_full))
        if not same:
            print("  cont:", cont)
            print("  full:", full)
        history.append((text, full))
    print("all identical" if not bad else "%d turn(s) differ" % bad)
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
