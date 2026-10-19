"""First-chunk policy for the TTS: time to first audio vs silent gaps, with the Nano's real TTS times.

Each reply is cut by a policy, every chunk is synthesised in order (as run_turn does), and playback
is simulated: chunk k starts when it is ready and the previous one has finished. Reported: first
audio (synthesis of chunk 1) and the total silence inserted between chunks. LLM token time is left
out (decoding a few tokens is ~0.15 s), so this isolates the TTS side.

  venv/bin/python -u app/tts_cut_check.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import find_cut, first_cut  # noqa: E402
from tts import Tts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPLIES = ["沒問題，請問我。", "台南是台灣南部的一個城市。", "帶小女兒走路學眼神無助，建議多觀察她的反應。",
           "媽媽聽了這話，會覺得你很愛她嗎？", "好，開始測試。", "你好！我隨時可以幫你。",
           "三月十五第二百九十九元折扣已確認。謝謝。", "去試吧，我隨時待著。"]


def chunks(text, first_min, later_min):
    out, buf = [], text
    while buf:
        if first_min is None and not out:
            cut = first_cut(buf, final=True)
        else:
            cut = find_cut(buf, first_min if not out else later_min)
        if cut < 0:
            out.append(buf)
            break
        out.append(buf[:cut])
        buf = buf[cut:]
    return [c for c in out if c.strip()]


def main():
    tts = Tts(os.environ.get("PRIME_ONNX", os.path.expanduser("~/primetts_v2/primetts_v21_3voice.onnx")),
              str(ROOT / "models" / "g2pw" / "g2pw_int8.onnx"), str(ROOT / "tts_res"), threads=4)
    tts.synth_sentence("暖機。")
    cache = {}

    def synth(c):
        if c not in cache:
            wav, dt = tts.synth_sentence(c)
            cache[c] = (dt, len(wav) / 16000)
        return cache[c]

    for name, first_min, later_min in (("current (comma after 6)", 6, 6), ("first comma after 2", 2, 6),
                                       ("first_cut (word boundary)", None, 6)):
        firsts, gaps = [], []
        for r in REPLIES:
            t = 0.0  # synthesis clock
            play_end = None
            gap = 0.0
            parts = chunks(r, first_min, later_min)
            if first_min is None:
                print("   ", " | ".join(parts))
            for i, c in enumerate(parts):
                dt, dur = synth(c)
                t += dt
                if i == 0:
                    firsts.append(t)
                    play_end = t + dur
                else:
                    start = max(t, play_end)
                    gap += start - play_end
                    play_end = start + dur
            gaps.append(gap)
        print("%-26s first audio mean %.2fs max %.2fs | silence between chunks mean %.2fs max %.2fs"
              % (name, sum(firsts) / len(firsts), max(firsts), sum(gaps) / len(gaps), max(gaps)))


if __name__ == "__main__":
    main()
