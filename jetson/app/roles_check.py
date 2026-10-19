"""Does the model (VC_LLM) hold each role? Plays 3 scripted user lines per role and flags replies that break
character (mentions being an AI), use Markdown, run long, or reuse the same reply. Needs the GPU to
itself (stop the server first).

  venv/bin/python -u app/roles_check.py ./fwd35_serve [role_id ...]
"""
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import Llm  # noqa: E402
import roles  # noqa: E402
from server import runaway  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = {
    "support": ["我家的網路一直斷線", "指示燈是紅色的", "好，我重開機了，還是不行"],
}
BAD = re.compile(r"人工智慧|語言模型|語音助理|AI|我是機器|[*#`]|^\s*[-\d]+[.、)]")


def main(binary, ids):
    llm = Llm(binary, str(ROOT / "models" / os.environ.get("VC_LLM", "Ternary-Bonsai-2-2B-PTQ1_0.gguf")),
              str(ROOT / "models" / "tok" / "tokenizer.json"), max_new=80)
    flagged = 0
    for rid in ids:
        role = roles.BY_ID[rid]
        print("== %s (%s, voice %d)" % (role["name"], rid, role["voice"]))
        if role["opening"]:
            print("   [開場] %s" % role["opening"])
        history = []
        for u in SCRIPTS[rid]:
            t0 = time.monotonic()
            reply = "".join(llm.stream(history, u, roles.system_prompt(role))).strip()
            raw = reply
            bad = runaway(reply)
            reply = reply[:bad].strip() if bad >= 0 else reply  # what the server keeps
            dt = time.monotonic() - t0
            sentences = len(re.findall(r"[。！？!?]", reply)) or 1
            flags = []
            if BAD.search(reply) and not (rid == "assistant" and "助理" in reply):
                flags.append("out of character/markdown")
            if sentences > 3 or len(reply) > 70:
                flags.append("long")
            if any(reply == r for _, r in history):
                flags.append("repeat")
            flagged += bool(flags)
            if bad >= 0:
                flags.append("guard cut %d of %d chars" % (len(raw) - len(reply), len(raw)))
            print("   你：%s\n   答：%s   (%.1fs, %s)" % (u, reply, dt, ", ".join(flags) or "ok"))
            history.append((u, reply))
    print("%d flagged reply(ies)" % flagged)
    llm.proc.terminate()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or [r["id"] for r in roles.ROLES])
