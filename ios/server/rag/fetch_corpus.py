"""Download 臺灣通史 (連橫, 1918, public domain) from zh.wikisource as plain text chapters.

Usage: python fetch_corpus.py <out_dir> [<wikisource title>]
Writes one UTF-8 .txt per chapter link found on the index page. Polite: one request per second.
Default title: 臺灣通史.
"""

from __future__ import annotations

import html
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://zh.wikisource.org/wiki/"
UA = "voice-chat-rag-corpus/0.1 (public-domain text, personal demo)"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def chapter_text(page_html: str) -> str:
    body = page_html.split('id="mw-content-text"', 1)[-1]
    body = body.split('id="catlinks"', 1)[0]
    paras = re.findall(r"<p[^>]*>(.*?)</p>", body, flags=re.S)
    lines = []
    for p in paras:
        text = re.sub(r"<[^>]+>", "", p)
        text = html.unescape(text).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def main() -> None:
    out = Path(sys.argv[1])
    title = sys.argv[2] if len(sys.argv) > 2 else "臺灣通史"
    out.mkdir(parents=True, exist_ok=True)
    index = fetch(BASE + urllib.parse.quote(title))
    prefix = re.escape(urllib.parse.quote(title + "/"))
    chapters = []
    for href in re.findall(rf'href="/wiki/({prefix}[^"#?]+)"', index):
        if href not in chapters:
            chapters.append(href)
    print(f"{len(chapters)} chapters")
    for i, href in enumerate(chapters, 1):
        name = urllib.parse.unquote(href.split("/")[-1])
        text = chapter_text(fetch(BASE + href))
        (out / f"{i:02d}_{name}.txt").write_text(text, encoding="utf-8")
        print(f"{name}: {len(text)} chars")
        time.sleep(1)


if __name__ == "__main__":
    main()
