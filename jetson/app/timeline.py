"""Timeline of what each AI stage does and what the Jetson's CPU cores / GPU are doing meanwhile.

Spans (component, start, end, label) are recorded around VAD, ASR, LLM, TTS and network calls;
a sampler thread reads per-core CPU load (/proc/stat), GPU load (/sys/devices/gpu.0/load, 0..1000)
and used RAM (shared by CPU and GPU on the Nano) every SAMPLE_S. Everything older than KEEP_S is
dropped. `snapshot(seconds)` returns the window shown by web/trace.html.
"""
import threading
import time
from collections import deque
from contextlib import contextmanager

SAMPLE_S = 0.1
KEEP_S = 180
GPU_LOAD = "/sys/devices/gpu.0/load"

_spans = deque()
_marks = deque()
_samples = deque()
_lock = threading.Lock()


def _trim(q, now):
    while q and q[0][0] < now - KEEP_S:
        q.popleft()


def add_span(comp, t0, t1, label=""):
    with _lock:
        _spans.append((t0, t1, comp, label))
        _trim(_spans, t1)


@contextmanager
def span(comp, label=""):
    t0 = time.monotonic()
    try:
        yield
    finally:
        add_span(comp, t0, time.monotonic(), label)


def mark(name, label=""):
    """Instant event: speech_start, speech_end, user text, first token, played..."""
    now = time.monotonic()
    with _lock:
        _marks.append((now, name, label))
        _trim(_marks, now)


def _cpu_times():
    with open("/proc/stat") as f:
        rows = [ln.split() for ln in f if ln.startswith("cpu") and ln[3].isdigit()]
    return [(sum(map(int, r[1:])), int(r[4]) + int(r[5])) for r in rows]  # (total, idle+iowait)


def _gpu():
    try:
        with open(GPU_LOAD) as f:
            return int(f.read().strip()) / 10.0  # percent
    except OSError:
        return None


def _mem_used_mb():
    info = {}
    with open("/proc/meminfo") as f:
        for ln in f:
            k, v = ln.split(":")
            info[k] = int(v.split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) // 1024, info["MemTotal"] // 1024


def _sampler():
    prev = _cpu_times()
    while True:
        time.sleep(SAMPLE_S)
        cur = _cpu_times()
        cores = [round(100.0 * (1 - (ci - pi) / max(1, ct - pt)), 1)
                 for (ct, ci), (pt, pi) in zip(cur, prev, strict=True)]
        prev = cur
        now = time.monotonic()
        used, _ = _mem_used_mb()
        with _lock:
            _samples.append((now, cores, _gpu(), used))
            _trim(_samples, now)


def start():
    threading.Thread(target=_sampler, daemon=True, name="trace-sampler").start()


def snapshot(seconds=30):
    now = time.monotonic()
    t0 = now - seconds
    with _lock:
        spans = [s for s in _spans if s[1] >= t0]
        marks = [m for m in _marks if m[0] >= t0]
        samples = [s for s in _samples if s[0] >= t0]
    return {
        "now": now, "window": seconds, "mem_total_mb": _mem_used_mb()[1],
        "spans": [{"t0": a - now, "t1": b - now, "comp": c, "label": lb} for a, b, c, lb in spans],
        "marks": [{"t": t - now, "name": n, "label": lb} for t, n, lb in marks],
        "samples": [{"t": t - now, "cpu": c, "gpu": g, "mem": m} for t, c, g, m in samples],
    }
