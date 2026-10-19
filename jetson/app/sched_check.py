"""CPU scheduling experiments on the Nano: TTS thread count and core pinning, with the LLM engine decoding.

The engine (fwd35_serve) drives the GPU from one CPU thread; TTS (onnxruntime, intra-op threads) runs in
the same turn while the LLM is still decoding. Each config pins the engine and the TTS threads to core
sets (sched_setaffinity; onnxruntime threads inherit the affinity of the thread that creates the session)
and reports, with the LLM decoding in the background: mean synthesis time per sentence and LLM decode rate.
Needs the CPU and GPU to itself (stop the server first).

  venv/bin/python -u app/sched_check.py ./fwd35_serve
"""
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import Llm  # noqa: E402
from tts import Tts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ALL = {0, 1, 2, 3}
SENTS = ["好，收到。", "台南是台灣南部的一個城市。", "帶小女兒走路學眼神無助，建議多觀察她的反應。",
         "媽媽聽了這話，會覺得你很愛她嗎？", "你好！我隨時可以幫你。", "三月十五第二百九十九元折扣已確認。"]
# (label, tts threads, tts cores, engine cores)
CONFIGS = [("T4 free / engine free (current)", 4, ALL, ALL), ("T3 free / engine free", 3, ALL, ALL),
           ("T2 free / engine free", 2, ALL, ALL), ("T3 on 1-3 / engine on 0", 3, {1, 2, 3}, {0}),
           ("T4 free / engine on 0", 4, ALL, {0}), ("T2 on 1-2 / engine on 0", 2, {1, 2}, {0})]


def cpu_s(pid):
    f = open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def run(llm, label, threads, tts_cores, eng_cores, prompts):
    os.sched_setaffinity(llm.proc.pid, eng_cores)
    os.sched_setaffinity(0, tts_cores)  # inherited by the onnxruntime pool created next
    tts = Tts(os.environ.get("PRIME_ONNX", os.path.expanduser("~/primetts_v2/primetts_v21_3voice.onnx")),
              str(ROOT / "models" / "g2pw" / "g2pw_int8.onnx"), str(ROOT / "tts_res"), threads=threads)
    tts.synth_sentence("暖機。")
    res = {}
    for mode in ("tts alone", "with LLM decoding"):
        stop, n, t_llm = threading.Event(), [0], [0.0]

        def bg():
            i = 0
            while not stop.is_set():
                t0 = time.monotonic()
                for _ in llm.stream([], prompts[i % len(prompts)]):
                    n[0] += 1
                    if stop.is_set():
                        llm.cancel()
                t_llm[0] += time.monotonic() - t0
                i += 1
        if mode != "tts alone":
            th = threading.Thread(target=bg)
            th.start()
            time.sleep(1.0)  # past the prefill
        c0, w0, n0 = cpu_s(llm.proc.pid), time.monotonic(), n[0]
        dts = [tts.synth_sentence(s)[1] for s in SENTS]
        wall, cpu = time.monotonic() - w0, cpu_s(llm.proc.pid) - c0
        rate = (n[0] - n0) / wall
        stop.set()
        if mode != "tts alone":
            th.join()
        res[mode] = (sum(dts) / len(dts), max(dts), rate, cpu / wall)
    a, b = res["tts alone"], res["with LLM decoding"]
    print("%-32s alone %.2fs | with LLM mean %.2fs max %.2fs (+%d%%) | LLM %.1f deltas/s, engine CPU %.2f core"
          % (label, a[0], b[0], b[1], round(100 * (b[0] / a[0] - 1)), b[2], b[3]), flush=True)
    del tts


def main(binary):
    llm = Llm(binary, str(ROOT / "models" / "Ternary-Bonsai-2-2B-PTQ1_0.gguf"),
              str(ROOT / "models" / "tok" / "tokenizer.json"), max_new=60)
    prompts = ["請介紹一下台灣的夜市文化", "請說明如何煮一碗好吃的牛肉麵", "給我一個週末旅行的建議"]
    for _ in llm.stream([], prompts[0]):
        pass  # warm the engine
    for cfg in CONFIGS:
        run(llm, *cfg, prompts)
    llm.proc.terminate()


if __name__ == "__main__":
    main(sys.argv[1])
