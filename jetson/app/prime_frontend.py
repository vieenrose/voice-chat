"""PrimeTTS text frontend (zh-TW + English) for the Jetson demo.

A line-by-line port of the iOS app's Swift frontend (Local/TTS/TextNorm.swift, NumberWords.swift,
PrimeFrontend.swift, G2PW.swift, G2PEn.swift), which itself ports PrimeTTS `frontend_bopomofo.py`.
Output is verified against the iOS golden file (checks/FrontendCheck/golden.json) by check_golden.py.
"""

import re
import unicodedata

import numpy as np

PUNCT = {",", ".", "?", "!", "…", "-", "'"}
ADDR = set("號樓段巷弄室坪")
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
ZH_DIGITS = "零一二三四五六七八九"
EN_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


# --------------------------------------------------------------------------- number words

def digits_zh(digits):
    return "".join(ZH_DIGITS[int(c)] for c in digits if c.isdigit())


def digits_en(digits):
    return " ".join(EN_DIGITS[int(c)] for c in digits if c.isdigit())


def _group_zh(n):
    d = [n // 1000, n // 100 % 10, n // 10 % 10, n % 10]
    units = ["千", "百", "十", ""]
    out, pending_zero = "", False
    for i, v in enumerate(d):
        if v == 0:
            if out:
                pending_zero = True
            continue
        if pending_zero:
            out += "零"
            pending_zero = False
        out += ZH_DIGITS[v] + units[i]
    return out


def cardinal_zh(value):
    """cn2an.an2cn(n, "low"): 10 -> 十, 110 -> 一百一十, 10001 -> 一萬零一 (亿/万)."""
    value = int(value)
    if value == 0:
        return "零"
    yi, wan, ge = value // 100_000_000, value // 10_000 % 10_000, value % 10_000
    parts = []
    if yi > 0:
        parts.append((yi, "亿"))
    if wan > 0:
        parts.append((wan, "万"))
    if ge > 0:
        parts.append((ge, ""))
    out = ""
    for i, (p, unit) in enumerate(parts):
        text = _group_zh(p)
        if i > 0 and p < 1000:
            text = "零" + text
        out += text + unit
    if out.startswith("一十"):
        out = out[1:]
    return out


def cardinal_zh_str(digits):
    return cardinal_zh(int(digits)) if digits.isdigit() else digits_zh(digits)


_SMALL = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _below_1000(n, and_word=False):
    words = []
    if n >= 100:
        words += [_SMALL[n // 100], "hundred"]
    r = n % 100
    if and_word and n >= 100 and r > 0:
        words.append("and")
    if r >= 20:
        words.append(_TENS[r // 10])
        if r % 10 > 0:
            words.append(_SMALL[r % 10])
    elif r > 0:
        words.append(_SMALL[r])
    return words


def cardinal_en(value, and_word=False):
    value = int(value)
    if value == 0:
        return "zero"
    words, rest = [], value
    for size, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")):
        if rest >= size:
            words += _below_1000(rest // size, and_word) + [name]
            rest %= size
    if and_word and 0 < rest < 100 and value >= 100:
        words.append("and")
    words += _below_1000(rest, and_word)
    return " ".join(words)


def cardinal_en_str(digits):
    return cardinal_en(int(digits)) if digits.isdigit() else digits_en(digits)


_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
              "nine": "ninth", "twelve": "twelfth"}


def ordinal_en(value):
    words = cardinal_en(value, and_word=True).split(" ")
    last = words.pop() if words else ""
    if not last:
        return ""
    if last in _IRREGULAR:
        word = _IRREGULAR[last]
    elif last.endswith("y"):
        word = last[:-1] + "ieth"
    else:
        word = last + "th"
    return " ".join(words + [word])


def ordinal_en_str(digits):
    return ordinal_en(int(digits)) if digits.isdigit() else digits_en(digits)


# --------------------------------------------------------------------------- text normalizer

def _is_zh(c):
    return 0x4E00 <= ord(c) <= 0x9FFF


def _is_latin(c):
    return c.isascii() and c.isalpha()


def _en_context(text, s, e):
    """True when the nearest letter/CJK neighbour of text[s:e] is Latin."""
    left = next((c for c in reversed(text[:s]) if _is_zh(c) or _is_latin(c)), None)
    right = next((c for c in text[e:] if _is_zh(c) or _is_latin(c)), None)
    for c in (left, right):
        if c is None:
            continue
        if _is_latin(c):
            return True
        if _is_zh(c):
            return False
    return False


def _sub(pattern, text, repl, flags=0):
    """re.sub with a callback that gets (match, groups) where unmatched groups are None."""
    def cb(m):
        return repl(m)
    return re.sub(pattern, cb, text, flags=flags)


def _year_en(y):
    if 2000 <= y <= 2009:
        return "two thousand" + ("" if y == 2000 else " " + cardinal_en(y % 100))
    if 1000 <= y <= 2099:
        r = y % 100
        tail = ("oh " + cardinal_en(r)) if 0 < r < 10 else (cardinal_en(r) if r > 0 else "hundred")
        return cardinal_en(y // 100) + " " + tail
    return cardinal_en(y)


def _collapse(s):
    return " ".join(s.split())


def normalize(text):
    if not text:
        return text

    # 1) email -> spelled
    def email(m):
        s = m.group(0).replace("@", " at ").replace(".", " dot ")
        s = re.sub(r"\d", lambda d: " " + digits_en(d.group(0)) + " ", s)
        return " " + _collapse(s) + " "
    text = re.sub(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", email, text)

    # 1.5) time of day
    t0 = text

    def clock(m):
        h, mm, ap = int(m.group(1)), m.group(2), m.group(3)
        mi = int(mm)
        is_am = ap is not None and "a" in ap.lower()
        if _en_context(t0, m.start(), m.end()):
            hh = cardinal_en(h)
            t = (hh + " o'clock") if mi == 0 else ((hh + " oh " + cardinal_en(mm)) if mi < 10 else (hh + " " + cardinal_en(mm)))
            if ap is not None:
                t += " a m" if is_am else " p m"
            return " " + t + " "
        pre = "" if ap is None else ("上午" if is_am else "下午")
        tail = "整" if mi == 0 else ("半" if mi == 30 else cardinal_zh(mi) + "分")
        return " " + pre + cardinal_zh(h) + "點" + tail + " "
    text = re.sub(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*([AaPp][.]?[Mm][.]?)?\b", clock, t0)

    # 2) date
    def zh_date(m):
        out = digits_zh(m.group(1)) + "年"
        if m.group(2):
            out += cardinal_zh_str(m.group(2)) + "月"
        if m.group(3):
            out += cardinal_zh_str(m.group(3)) + "日"
        return out
    text = re.sub(r"(\d{4})\s*年\s*(?:(\d{1,2})\s*月)?\s*(?:(\d{1,2})\s*[日號])?", zh_date, text)

    def en_date(m):
        out = m.group(1) + " " + ordinal_en_str(m.group(2))
        if m.group(3) and m.group(3).isdigit():
            out += " " + _year_en(int(m.group(3)))
        return out
    text = re.sub(r"\b(" + MONTHS + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b", en_date, text)

    # 3) temperature
    t3 = text

    def temp(m):
        neg = m.group(1) in ("-", "－")
        n = m.group(2)
        if _en_context(t3, m.start(), m.end()):
            return " " + ("minus " if neg else "") + cardinal_en_str(n) + " degrees Celsius "
        return " 攝氏" + ("零下" if neg else "") + cardinal_zh_str(n) + "度 "
    text = re.sub(r"(?:攝氏)?\s*([-－]?)(\d+)\s*(?:°\s*[Cc]|度[Cc]?|℃)", temp, t3)

    # 4) percent
    t4 = text

    def percent(m):
        n = m.group(1)
        if _en_context(t4, m.start(), m.end()):
            return " " + cardinal_en_str(n) + " percent "
        return " 百分之" + cardinal_zh_str(n) + " "
    text = re.sub(r"(\d+)\s*[%％]", percent, t4)

    # 5) price
    t5 = text

    def price(m):
        whole = m.group(1).replace(",", "")
        cents = m.group(2)
        en = _en_context(t5, m.start(), m.end()) or "USD" in m.group(0) or "US$" in m.group(0)
        if en:
            out = cardinal_en_str(whole) + " dollar" + ("" if whole == "1" else "s")
            if cents:
                out += " and " + cardinal_en_str(cents) + " cent" + ("" if cents == "01" else "s")
        else:
            out = cardinal_zh_str(whole) + "元"
            if cents:
                out += cardinal_zh_str(cents) + "分"
        return " " + out + " "
    text = re.sub(r"(?:NT\$|US\$|USD|\$)\s*([\d,]+)(?:\.(\d{2}))?", price, t5)
    text = re.sub(r"([\d,]+)\s*(元|塊錢|塊|台幣|新台幣)",
                  lambda m: " " + cardinal_zh_str(m.group(1).replace(",", "")) + (m.group(2) or "元") + " ", text)

    # 6) phone / extension groups -> digit by digit
    t6 = text

    def phone(m):
        en = _en_context(t6, m.start(), m.end())
        trimmed = m.group(0).strip("()")
        groups = [g for g in re.split(r"[-\s]+", trimmed) if g]
        groups = ["".join(c for c in g if c.isnumeric()) for g in groups]
        return " " + " ".join(digits_en(g) if en else digits_zh(g) for g in groups if g) + " "
    text = re.sub(r"\(?\d{2,4}\)?(?:[-\s]\d{2,4}){1,4}", phone, t6)
    t6b = text

    def ext(m):
        return m.group(1) + " " + (digits_en(m.group(2)) if _en_context(t6b, m.start(), m.end()) else digits_zh(m.group(2)))
    text = re.sub(r"(分機|內線|ext\.?|extension)\s*(\d{2,6})", ext, t6b, flags=re.IGNORECASE)

    # 7) serial / order code
    def serial(m):
        out = []
        for c in m.group(0):
            if c.isalpha() or c.isnumeric():
                out.append(digits_en(c) if c.isnumeric() else c.upper())
        return " " + " ".join(out) + " "
    text = re.sub(r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9]{2,}(?:-[A-Za-z0-9]+)*\b", serial, text)

    # 7.5) decimals
    t75 = text

    def decimal(m):
        whole, frac = m.group(1), m.group(2)
        if _en_context(t75, m.start(), m.end()):
            return " " + cardinal_en_str(whole) + " point " + digits_en(frac) + " "
        return " " + cardinal_zh_str(whole) + "點" + digits_zh(frac) + " "
    text = re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", decimal, t75)

    # 8) counts / remaining digit runs
    t8 = text

    def count(m):
        d = m.group(0)
        en = _en_context(t8, m.start(), m.end())
        after = t8[m.end()] if m.end() < len(t8) else None
        if len(d) >= 5:
            return digits_en(d) if en else digits_zh(d)
        if en:
            return digits_en(d) if len(d) >= 4 else d
        # Python quirk kept: at end of string the iOS port treats the missing neighbour as an address mark
        if len(d) >= 4 and not (after is None or after in ADDR):
            return digits_zh(d)
        return cardinal_zh(int(d))
    text = re.sub(r"\d+", count, text)

    for a, b in (("，", ","), ("。", "."), ("？", "?"), ("！", "!")):
        text = text.replace(a, b)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.?!])", lambda m: m.group(1), text)
    return text.strip()


# --------------------------------------------------------------------------- bopomofo (g2pw)

class BertTokenizer:
    CLS, SEP, UNK = 101, 102, 100

    def __init__(self, vocab_text):
        self.vocab = {}
        for i, line in enumerate(vocab_text.split("\n")):
            if line:
                self.vocab[line] = i

    def id(self, token):
        return self.vocab.get(token, self.UNK)

    def _wordpiece(self, word):
        if word in self.vocab:
            return [word]
        pieces, chars, start = [], list(word), 0
        while start < len(chars):
            end, found = len(chars), None
            while start < end:
                sub = ("##" if start > 0 else "") + "".join(chars[start:end])
                if sub in self.vocab:
                    found = sub
                    break
                end -= 1
            if found is None:
                return ["[UNK]"]
            pieces.append(found)
            start = end
        return pieces

    @staticmethod
    def _strip_accents(s):
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    def tokenize_and_map(self, chars):
        def is_word_char(c):
            return c.isascii() and c.isalnum()

        words, text_to_word = [], [None] * len(chars)
        i = 0
        while i < len(chars):
            c = chars[i]
            if c == " ":
                i += 1
                continue
            j = i + 1
            if is_word_char(c):
                while j < len(chars) and is_word_char(chars[j]):
                    j += 1
            for k in range(i, j):
                text_to_word[k] = len(words)
            words.append(("".join(chars[i:j]), i, j))
            i = j

        tokens, spans = [], []
        for word, s, e in words:
            pieces = self._wordpiece(self._strip_accents(word).lower())
            if not pieces or pieces == ["[UNK]"]:
                spans.append((s, e))
                tokens.append("[UNK]")
            else:
                cursor = s
                for p in pieces:
                    length = len(p) - 2 if p.startswith("##") else len(p)
                    spans.append((cursor, cursor + length))
                    tokens.append(p)
                    cursor += length
        mapping = list(text_to_word)
        for t, (a, b) in enumerate(spans):
            for k in range(a, min(b, len(chars))):
                mapping[k] = t
        return tokens, mapping


class G2PW:
    """Per-character bopomofo for one sentence; polyphones are scored by the g2pw ONNX model."""
    WINDOW = 32

    def __init__(self, resources, vocab_text):
        self.labels = resources["labels"]
        self.char2ph = resources["char2ph"]
        self.single = resources["single"]
        self.chars = resources["chars"]
        self.char_index = {c: i for i, c in enumerate(self.chars)}
        self.tokenizer = BertTokenizer(vocab_text)

    def _make_query(self, chars, index):
        start, end = max(0, index - self.WINDOW // 2), min(len(chars), index + self.WINDOW // 2)
        text = [c.lower() for c in chars[start:end]]
        q = index - start
        tokens, text2token = self.tokenizer.tokenize_and_map(text)
        if q >= len(text2token) or text2token[q] is None:
            return None
        ids = [BertTokenizer.CLS] + [self.tokenizer.id(t) for t in tokens] + [BertTokenizer.SEP]
        ch = chars[index]
        if ch not in self.char_index or ch not in self.char2ph:
            return None
        return {"index": index, "ids": ids, "position": text2token[q] + 1,
                "char_id": self.char_index[ch], "allowed": self.char2ph[ch]}

    def convert(self, sentence, score):
        chars = list(sentence)
        result = [None] * len(chars)
        queries = []
        for i, ch in enumerate(chars):
            if ch in self.char_index:
                q = self._make_query(chars, i)
                if q is not None:
                    queries.append(q)
            elif ch in self.single:
                result[i] = self.single[ch]
        if not queries:
            return result
        size = len(self.labels)
        width = max(len(q["ids"]) for q in queries)
        ids = [q["ids"] + [0] * (width - len(q["ids"])) for q in queries]
        masks = []
        for q in queries:
            m = [0.0] * size
            for k in q["allowed"]:
                m[k] = 1.0
            masks.append(m)
        probs = score(ids, [q["position"] for q in queries], [q["char_id"] for q in queries], masks)
        for q, row in zip(queries, probs):
            best = int(np.argmax(row))
            result[q["index"]] = self.labels[best]
        return result


# --------------------------------------------------------------------------- English g2p

_GRAPHEMES = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
_PHONEMES = ["<pad>", "<unk>", "<s>", "</s>"] + (
    "AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 "
    "ER0 ER1 ER2 EY0 EY1 EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S "
    "SH T TH UH0 UH1 UH2 UW UW0 UW1 UW2 V W Y Z ZH").split()


class G2PEn:
    """CMUdict lookup, then the g2p_en GRU seq2seq (greedy) for unknown words."""

    def __init__(self, cmudict_text, weights_path):
        self.cmu = {}
        for line in cmudict_text.split("\n"):
            parts = [p for p in line.split(" ") if p]
            if len(parts) > 1:
                self.cmu[parts[0]] = parts[1:]
        self.net = _G2PNet(np.fromfile(weights_path, dtype="<f4"))

    def pronounce(self, word):
        w = word.lower()
        if w in self.cmu:
            return self.cmu[w]
        return self.net.predict(w)


class _G2PNet:
    H = 256

    def __init__(self, all_w):
        sizes = [29 * 256, 768 * 256, 768 * 256, 768, 768, 74 * 256, 768 * 256, 768 * 256, 768, 768, 74 * 256, 74]
        if all_w.size != sum(sizes):
            raise ValueError("g2pen.bin has %d floats, expected %d" % (all_w.size, sum(sizes)))
        parts, o = [], 0
        for n in sizes:
            parts.append(all_w[o:o + n])
            o += n
        (self.enc_emb, self.enc_wih, self.enc_whh, self.enc_bih, self.enc_bhh,
         self.dec_emb, self.dec_wih, self.dec_whh, self.dec_bih, self.dec_bhh,
         self.fc_w, self.fc_b) = parts
        self.enc_emb = self.enc_emb.reshape(29, 256)
        self.dec_emb = self.dec_emb.reshape(74, 256)
        self.enc_wih, self.enc_whh = self.enc_wih.reshape(768, 256), self.enc_whh.reshape(768, 256)
        self.dec_wih, self.dec_whh = self.dec_wih.reshape(768, 256), self.dec_whh.reshape(768, 256)
        self.fc_w = self.fc_w.reshape(74, 256)

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _cell(self, x, h_prev, wih, whh, bih, bhh):
        gi = wih @ x + bih
        gh = whh @ h_prev + bhh
        h = self.H
        r = self._sig(gi[:h] + gh[:h])
        z = self._sig(gi[h:2 * h] + gh[h:2 * h])
        n = np.tanh(gi[2 * h:] + r * gh[2 * h:])
        return (1 - z) * n + z * h_prev

    def predict(self, word):
        ids = [_GRAPHEMES.index(c) if c in _GRAPHEMES else 1 for c in word] + [2]
        hidden = np.zeros(self.H, dtype=np.float32)
        for i in ids:
            hidden = self._cell(self.enc_emb[i], hidden, self.enc_wih, self.enc_whh, self.enc_bih, self.enc_bhh)
        token, out = 2, []
        for _ in range(20):
            hidden = self._cell(self.dec_emb[token], hidden, self.dec_wih, self.dec_whh, self.dec_bih, self.dec_bhh)
            logits = self.fc_w @ hidden + self.fc_b
            best = int(np.argmax(logits))
            if best == 3:
                break
            out.append(_PHONEMES[best])
            token = best
        return out


# --------------------------------------------------------------------------- frontend

class PrimeFrontend:
    """zh-TW/en text -> PrimeTTS inputs: phone ids, tones and per-phone language (0 zh, 1 en)."""

    def __init__(self, symbols, g2pw, g2pen, score):
        self.symbol_ids = {s: i for i, s in enumerate(symbols)}
        self.g2pw = g2pw
        self.g2pen = g2pen
        self.score = score

    def _split(self, syllable):
        s = list(syllable)
        tone = 0
        if s and s[-1].isdigit():
            tone = int(s.pop())
        units = [u for u in s if u in self.symbol_ids]
        if len(units) == 1 and units[0] in "ㄓㄔㄕㄖㄗㄘㄙ" and "ㄭ" in self.symbol_ids:
            units.append("ㄭ")
        return units, tone

    def convert(self, raw):
        text = normalize(raw)
        bopomofo = self.g2pw.convert(text, self.score)
        chars = list(text)
        out = {"phones": [], "phone_ids": [], "tones": [], "langs": []}

        def add(phone, tone, lang):
            out["phones"].append(phone)
            out["phone_ids"].append(self.symbol_ids.get(phone, self.symbol_ids.get("UNK", 2)))
            out["tones"].append(tone)
            out["langs"].append(lang)

        i = 0
        while i < len(chars):
            ch = chars[i]
            if i < len(bopomofo) and bopomofo[i] is not None:
                units, tone = self._split(bopomofo[i])
                for u in units:
                    add(u, min(tone, 5), 0)
                i += 1
            elif ch.isascii() and ch.isalpha():
                j = i
                while j < len(chars) and chars[j].isascii() and (chars[j].isalpha() or chars[j] == "'"):
                    j += 1
                for p in self.g2pen.pronounce("".join(chars[i:j])):
                    phone, stress = p, 0
                    if phone and phone[-1].isdigit():
                        stress = int(phone[-1])
                        phone = phone[:-1]
                    if phone in self.symbol_ids:
                        add(phone, stress, 1)
                add("SP", 0, 1)
                i = j
            else:
                if ch in PUNCT:
                    add(ch, 0, 0)
                elif ch.isspace() and out["phones"] and out["phones"][-1] != "SP":
                    add("SP", 0, 0)
                i += 1
        return out


def blank_interleave(ids):
    """PrimeTTS add_blank: [0, a, 0, b, ..., 0]."""
    out = [0]
    for v in ids:
        out += [v, 0]
    return out
