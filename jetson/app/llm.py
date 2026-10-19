"""Persistent PULSAR engine wrapper (fwd35_serve) with Qwen chat prompts and barge-in.

The engine takes token ids and returns greedy ids, one per line, ended by `END`. This module owns
the subprocess, builds the Qwen3.5 chat prompt in non-thinking mode, detokenises incrementally
(a multi-byte character is held back until complete) and sends `STOP` to cut the current reply.

Multi-turn reuse: the engine keeps its K/V cache and recurrent state after a request and reports
how many tokens it holds ("P n"). When the next prompt starts with exactly those tokens, only the
rest is sent as a "+" continuation, so a turn prefills its new tokens instead of the whole history.
Any mismatch (a cut reply not kept in the history, the history window sliding) falls back to a
full prefill.
"""
import json
import os
import subprocess
import threading

from tokenizers import Tokenizer

SYSTEM = ("你是語音助理。請用台灣繁體中文，口語、簡短地回答，一到兩句話就好。"
          "不要使用 Markdown、列表、星號或表情符號。")
IM_START, IM_END, ENDOFTEXT = 151644, 151645, 151643
STOP_IDS = {IM_END, ENDOFTEXT}
MAX_CTX = 4096
HISTORY_MAX = 8  # history window: when it would exceed HISTORY_MAX turns it restarts at HISTORY_KEEP,
HISTORY_KEEP = 4  # so the prompt prefix (and the engine's state) stays valid for several turns


def build_prompt(tok, history, user_text, system=SYSTEM):
    """history: list of (user, assistant) pairs. Returns token ids."""
    parts = ["<|im_start|>system\n" + system + "<|im_end|>\n"]
    for u, a in history:
        parts.append("<|im_start|>user\n" + u + "<|im_end|>\n")
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n" + a + "<|im_end|>\n")
    parts.append("<|im_start|>user\n" + user_text + "<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return tok.encode("".join(parts), add_special_tokens=False).ids


class Llm:
    def __init__(self, binary, model, tok_json, max_new=200):
        self.tok = Tokenizer.from_file(tok_json)
        # the model sometimes writes a think tag itself; it is not special for the decoder and the TTS would read it out
        self.skip_ids = {i for i in (self.tok.token_to_id("<think>"), self.tok.token_to_id("</think>")) if i is not None}
        self.max_new = max_new
        self.proc = subprocess.Popen(
            [binary, model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, bufsize=1, cwd=os.path.dirname(binary) or None)
        self.ready = threading.Event()
        self.lock = threading.Lock()  # one request at a time; the engine has one decode stream
        self.held = []  # token ids in the engine's state after the last request
        self.reused = 0  # tokens skipped by the last request (for logs and tests)
        self.win = 0  # index of the first history turn in the prompt
        threading.Thread(target=self._drain_err, daemon=True).start()
        if not self.ready.wait(300):
            raise RuntimeError("fwd35_serve did not become ready")

    def _drain_err(self):
        for line in self.proc.stderr:
            if "ready" in line:
                self.ready.set()

    def cancel(self):
        """Barge-in: ends the decode in progress after the current token; safe to call anytime."""
        try:
            self.proc.stdin.write("STOP\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def _prefill_only(self, ids):
        """Feeds `ids` from scratch and keeps them as the engine state (one decoded token is discarded)."""
        self.held = []
        self.proc.stdin.write("1 %s\n" % " ".join(map(str, ids)))
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline().strip()
            if not line or line in ("END", "ERR"):
                break
            if line.startswith("P "):
                self.held = ids[:int(line[2:])]

    def prepare(self, history, system=SYSTEM):
        """If the next turn would restart the history window (its whole prefix is prefilled again, ~45 ms per
        token on the 4B), prefill that prefix now, while the user listens and answers."""
        with self.lock:
            win = self.win if self.win <= len(history) else 0
            if len(history) - win <= HISTORY_MAX:
                return
            win = len(history) - HISTORY_KEEP
            text = "<|im_start|>system\n" + system + "<|im_end|>\n"
            for u, a in history[win:]:
                text += "<|im_start|>user\n" + u + "<|im_end|>\n"
                text += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + a + "<|im_end|>\n"
            ids = self.tok.encode(text, add_special_tokens=False).ids
            self.win = win
            if self.held[:len(ids)] != ids:
                self._prefill_only(ids)

    def warm(self, system=SYSTEM):
        """Prefills the system block ahead of the first turn (the Nano's 4B prefills ~45 ms per token), so
        that turn only prefills what the user said. A no-op if the engine state already starts with it."""
        ids = self.tok.encode("<|im_start|>system\n" + system + "<|im_end|>\n", add_special_tokens=False).ids
        with self.lock:
            if self.held[:len(ids)] == ids:
                return
            self._prefill_only(ids)

    def stream(self, history, user_text, system=SYSTEM):
        """Yields text deltas. Stops early after cancel(). Ends with the full reply in `self.last`."""
        if self.win > len(history):  # new session: the history restarted
            self.win = 0
        if len(history) - self.win > HISTORY_MAX:
            self.win = len(history) - HISTORY_KEEP
        ids = build_prompt(self.tok, history[self.win:], user_text, system)
        if len(ids) + self.max_new > MAX_CTX:
            raise ValueError("prompt too long: %d tokens" % len(ids))
        with self.lock:
            n = len(self.held)
            cont = 0 < n < len(ids) and ids[:n] == self.held
            send = ids[n:] if cont else ids
            self.reused = n if cont else 0
            base = self.held if cont else []
            self.held = []  # unknown until the engine reports its position
            self.proc.stdin.write("%s%d %s\n" % ("+" if cont else "", self.max_new, " ".join(map(str, send))))
            self.proc.stdin.flush()
            out, sent, emitted = [], "", []
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("fwd35_serve exited")
                line = line.strip()
                if line == "END":
                    break
                if line == "ERR":
                    raise RuntimeError("fwd35_serve rejected the request")
                if line.startswith("P "):
                    # processing is append-only: the state is a prefix of what was sent plus emitted
                    self.held = (base + send + emitted)[:int(line[2:])]
                    continue
                tid = int(line)
                emitted.append(tid)
                if tid in STOP_IDS:
                    continue  # keep reading until END; the engine stops after EOS anyway
                if tid in self.skip_ids:
                    continue
                out.append(tid)
                text = self.tok.decode(out, skip_special_tokens=True)
                if text.endswith("�"):
                    continue  # incomplete UTF-8 sequence, wait for the next token
                delta = text[len(sent):]
                sent = text
                if delta:
                    yield delta
            self.last = sent


if __name__ == "__main__":
    import sys
    import time

    llm = Llm(sys.argv[1], sys.argv[2], sys.argv[3])
    t0 = time.time()
    first = None
    for d in llm.stream([], sys.argv[4] if len(sys.argv) > 4 else "台灣的首都是哪裡？"):
        if first is None:
            first = time.time() - t0
        print(d, end="", flush=True)
    print()
    print(json.dumps({"first_delta_s": round(first or 0, 3), "total_s": round(time.time() - t0, 3)}))
