"""Download 聖經 和合本 (public domain) from zh.wikisource, one UTF-8 .txt per book.

Usage: python fetch_bible.py <out_dir>
Each book page is followed for chapter subpages when it has them; otherwise the book page is used.
Polite: one request per second.
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
ROOT = "聖經_(和合本)"
UA = "voice-chat-rag-corpus/0.1 (public-domain text, personal demo)"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def paragraphs(page_html: str) -> str:
    body = page_html.split('id="mw-content-text"', 1)[-1].split('id="catlinks"', 1)[0]
    lines = []
    for p in re.findall(r"<p[^>]*>(.*?)</p>", body, flags=re.S):
        text = html.unescape(re.sub(r"<[^>]+>", "", p)).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def book_links(index_html: str, parent: str = ROOT) -> list[str]:
    """Subpages of `parent` as raw (still percent-encoded) hrefs, in page order."""
    found: list[str] = []
    for href in re.findall(r'href="/wiki/([^"#?]+)"', index_html):
        if urllib.parse.unquote(href).startswith(parent + "/") and href not in found:
            found.append(href)
    return found


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    index = fetch(BASE + urllib.parse.quote(ROOT))
    books = book_links(index)
    print(f"{len(books)} books")
    for i, href in enumerate(books, 1):
        name = urllib.parse.unquote(href.split("/")[-1])
        pages = [href]
        time.sleep(1)
        book_html = fetch(BASE + href)
        # Chapter subpages (e.g. .../創世記/第1章) when the book page links to them.
        chapters = book_links(book_html, parent=urllib.parse.unquote(href))
        if chapters:
            pages = chapters
        texts = []
        for page in pages:
            if page != href:
                time.sleep(1)
                texts.append(paragraphs(fetch(BASE + page)))
            else:
                texts.append(paragraphs(book_html))
        text = "\n\n".join(t for t in texts if t)
        (out / f"{i:02d}_{name}.txt").write_text(text, encoding="utf-8")
        print(f"{name}: {len(pages)} pages, {len(text)} chars")


if __name__ == "__main__":
    main()
