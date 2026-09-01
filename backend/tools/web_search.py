"""
Web search tool — self-hosted SearXNG compatible.

Tries in order:
  1. Local SearXNG at `$SEARXNG_URL/search?format=json` (self-hosted)
  2. DuckDuckGo via duckduckgo_search.DDGS (pip)
  3. Lite DuckDuckGo scrape via httpx + lxml
  4. Generic offline placeholder (only when everything above is unreachable)

API matches SearXNG json: /search?q=...&format=json&categories=general
Also exposes `web_search(query, count=5)` async function for tool calling.

Uses caching (5 min TTL, LRU-bounded) to keep latency low.
"""
import asyncio
import os
import time
import re
import hashlib
import threading
import urllib.parse
from collections import OrderedDict
import httpx
from typing import List, Dict
from loguru import logger

# SearXNG base URL: resolved lazily through _searxng_base() so the Docker/compose
# value (SEARXNG_URL=http://searxng:8080) is actually honored by the search path
# itself. It used to be hard-coded to http://localhost:8888 here, which meant that
# in compose /health reported SearXNG "ok" (app.py read the env var) while every
# real search silently missed it and fell through to scraping.
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")


def _searxng_base() -> str:
    return (os.getenv("SEARXNG_URL") or SEARXNG_URL or "http://localhost:8888").rstrip("/")


# In-memory cache: LRU-bounded so a long-running demo can't grow it without limit
# (it was a plain dict keyed by query hash, evicted only by TTL, i.e. unbounded for
# a stream of unique queries). TTL checked on read; size checked on write.
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
CACHE_TTL = 300  # 5 min
CACHE_MAX = 256
_CACHE_LOCK = threading.Lock()

# No curated mock DB — honest search only; generic mock below is only for truly offline case

def _cache_get(key: str) -> dict | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if not entry:
            return None
        if time.time() - entry["t"] >= CACHE_TTL:
            _CACHE.pop(key, None)
            return None
        _CACHE.move_to_end(key)
        return entry["data"]


def _cache_put(key: str, data: dict) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = {"t": time.time(), "data": data}
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_MAX:
            _CACHE.popitem(last=False)


def _cache_clear() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _cache_key(query: str, count: int) -> str:
    return hashlib.md5(f"{query.lower().strip()}::{count}".encode()).hexdigest()

def format_results(results: List[Dict], max_chars=4000) -> str:
    """Format search results for LLM context (rich, <4000 chars) — keep snippets long for informative answers"""
    lines = []
    for i, r in enumerate(results[:5], 1):
        title = r.get("title","")[:150]
        url = r.get("url","")
        content = r.get("content","")[:600].replace("\n"," ").strip()
        if not content:
            content = r.get("title","")
        lines.append(f"[{i}] {title}\n    URL: {url}\n    Date/Snippet: {content}")
    txt = "\n".join(lines)
    return txt[:max_chars]

ADULT_KEYWORDS = ["bokep", "porn", "xxx", "xvideos", "xhamster", "bokepindoh", "avtub"]

def _is_adult(title: str, content: str, url: str) -> bool:
    t = (title + " " + content + " " + url).lower()
    return any(k in t for k in ADULT_KEYWORDS)

def _is_chinese(q: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in q)

def repair_truncated_json_query(s: str) -> str:
    """Best-effort repair for a tool-call query arg that arrives as JSON missing
    its closing brace(s), e.g. '{"query": "foo"'. This is not a hypothetical edge
    case: qwen_agent's own fncall_prompts/nous_fncall_prompt.py:extract_fn() treats
    ANY <tool_call> block that never got its closing </tool_call> tag as still
    mid-stream and defensively strips the last character \u2014 so when the small
    quantized model finishes the JSON payload but skips re-emitting the closing
    XML tag, an otherwise-valid {"query": "..."} loses its final '}' before it
    ever reaches our tools. Try re-closing with a few plausible suffixes before
    falling back to a regex pull, then to the raw string untouched."""
    import json as _json
    for suffix in ("", "}", '"}', '"}}', "}}"):
        try:
            d = _json.loads(s + suffix)
            if isinstance(d, dict) and str(d.get("query", "")).strip():
                return str(d["query"]).strip()
        except Exception:
            continue
    m = re.search(r'"query"\s*:\s*"([^"]*)', s)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return s

def _clean_query(q: str, max_words: int = 6) -> str:
    """Reformulate query: drop stopwords/questions, keep meaningful content words."""
    # zh: keep as-is (no space tokenization)
    if _is_chinese(q):
        return q[:40]
    stop = {"the","is","are","was","were","do","does","did","what","who","when","where","why","how","a","an","in","on","for","to","of","and","or","with","about","please","tell","me","some","any","can","you","today","current","latest","our"}
    words = [w for w in re.split(r"[^a-z0-9'\-]+", q.lower()) if w and w not in stop][:max_words]
    return " ".join(words) if words else q.strip()

async def _wttr_weather(query: str) -> Dict | None:
    """Direct weather forecast via wttr.in (no API key, zh+en). Returns a synthetic result dict
    with real forecast numbers for weather-intent queries, or None."""
    ql = query.lower()
    if not re.search(r"(天气|天氣|氣候|气温|氣溫|预报|预报|降雨|weather|forecast|temperature|雨|風|风|雪|晴|多雲|多云)", ql):
        return None
    zh = bool(re.search(r"[\u4e00-\u9fff]", query))
    # Generic: remove common question/weather words to isolate location (language-agnostic, not demo-specific)
    loc = re.sub(r"[，。！？,?!.]+", " ", query)
    loc = re.sub(r"(天气|天氣|气温|氣溫|预报|預報|降雨|如何|怎么样|怎樣|多少|度|呢|啊|吗)", " ", loc, flags=re.I)
    loc = re.sub(r"\b(weather|forecast|temperature|today|tomorrow|next|week|weekend|how|is|the|what|like|in|at|for|this|a|an|and|of|please|now|right|can|you|tell|me|search|find|about)\b", " ", loc, flags=re.I)
    loc = re.sub(r"\s+", " ", loc).strip()
    if _is_chinese(query):
        # For Chinese, take last remaining Chinese chunk as location
        m = re.findall(r"[\u4e00-\u9fff]{2,6}", loc)
        loc = m[-1] if m else loc.strip()
    else:
        # For English, take last remaining word
        parts = [p for p in loc.split() if len(p) >= 2]
        loc = parts[-1] if parts else loc
    loc = loc.strip("，。！？,?!. ")
    if not loc or len(loc) > 16:
        loc = (query[-6:] if zh else query.split()[-1]) if query else ""
    if not loc:
        return None
    try:
        lang = "zh" if zh else "en"
        logger.info(f"_wttr loc final '{loc}' -> quote '{urllib.parse.quote(loc)}'")
        url = f"https://wttr.in/{urllib.parse.quote(loc)}?format=j1&lang={lang}"
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as c:
            r = await c.get(url)
        if r.status_code != 200:
            return None
        j = r.json()
        days = j.get("weather", [])
        if not days:
            return None
        day = 2 if re.search(r"(後天|后天|大后天|day after tomorrow)", ql) else (1 if re.search(r"(明天|tomorrow|next day)", ql) else 0)
        di = days[min(day, len(days)-1)]
        desc = "、".join(d["value"] for d in (di.get("weatherDesc") or [{}]) if d.get("value"))
        t_min = di.get("mintempC") or di.get("mintempF")
        t_max = di.get("maxtempC") or di.get("maxtempF")
        ctx = di.get("current_condition") or j.get("current_condition") or [{}]
        hum = (ctx[0] if ctx else {}).get("humidity") or ""
        label = "後天" if day == 2 else ("明天" if day == 1 else "今天")
        content = (f"【wttr.in 天气】{loc} {label}：{desc or "天气"}，气温 {t_min}-{t_max}°C，湿度 {hum}%。"
                   f"（数据来源 wttr.in 实时预报，city={loc}）")
        logger.info(f"web_search wttr.in for '{query}' -> {loc} day={label} {t_min}-{t_max}C {desc}")
        return {"title": f"{loc} {label}天气预报", "url": f"https://wttr.in/{urllib.parse.quote(loc)}",
                "content": content, "score": 1.0, "engine": "wttr.in"}
    except Exception as e:
        logger.debug(f"wttr.in failed {e}")
        return None


_GENERIC_TOPIC_WORDS = {"最新","科技","新聞","頭條","財經","娛樂","體育","國際","產業","要聞","今日","報導","消息","新知","資訊"}

def _is_bare_topic_query(q: str) -> bool:
    """True for a Chinese query that's just a topic/category phrase with no
    location or named entity (e.g. 最新科技新聞), as opposed to one already
    naming a place or entity (e.g. 法國新聞, 日本科技新聞) that a region
    qualifier would only muddy."""
    if not _is_chinese(q):
        return False
    import jieba
    words = [w.strip() for w in jieba.cut(q) if w.strip()]
    remaining = [w for w in words if w not in _GENERIC_TOPIC_WORDS and len(w) >= 2]
    return len(remaining) == 0

def _entity_first_query(q: str) -> str:
    """Generic query reformulation — no hard-coded outlet/location lists."""
    base = _clean_query(q) or q.strip()[:40]
    # A bare topic-only Chinese query (no location/entity already in it)
    # searches noticeably better regionalized: this app defaults to zh-TW,
    # and the self-hosted SearXNG instance returns far more on-topic Taiwan
    # coverage for e.g. 台灣最新科技新聞 than the same unregionalized 最新科技新聞
    # (verified directly against the live instance). Skip when a location is
    # already present, or for weather (handled separately by _wttr_weather).
    if _is_bare_topic_query(q) and "台灣" not in q and "台湾" not in q:
        return [f"台灣{base}"]
    return [base]

async def _try_searxng(query: str, count: int, engine: str | None = None) -> List[Dict] | None:
    """Try local self-hosted SearXNG on 8888. engine=None -> aggregate; else single engine by name."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            # language: zh-TW for Chinese, EN otherwise (bing with language=all returns zh-biased junk).
            # zh-TW (not zh-CN) is this app's default region/phrasing for bare Chinese queries too —
            # not just ones that explicitly mention Taiwan — since zh-TW is the app's primary language.
            _lang = "zh-TW" if _is_chinese(query) else "en"
            # Generic news detection (no hard-coded outlet list) — use SearXNG news category for news queries
            _is_news = bool(re.search(r"(news|headlines|頭條|新聞|breaking|latest.*news|news.*today)", query.lower()))
            _cat = "news" if _is_news else "general"
            params = {"q": query, "format": "json", "categories": _cat, "pageno": 1,
                      "language": _lang, "safesearch": 1}
            if engine:
                params["engines"] = engine
            resp = await client.get(f"{_searxng_base()}/search", params=params)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results") or data.get("answers") or []
                out = []
                for r in results:
                    # Honest: skip mock-offline fallback from minimal server — trigger real fallback instead
                    if (r.get("engine") or "").startswith("mock"):
                        continue
                    title = r.get("title") or r.get("answer") or "Result"
                    url = r.get("url") or r.get("source") or ""
                    content = r.get("content") or r.get("answer") or ""
                    if _is_adult(title, content, url):
                        continue
                    # skip thin results (no content/title) — avoids junk like 'Meet app' for weather
                    if len(title.strip()) < 8 and len(content.strip()) < 20:
                        continue
                    out.append({"title": title, "url": url, "content": content,
                                "score": r.get("score", 0), "engine": r.get("engine", engine or "searxng")})
                    if len(out) >= count:
                        break
                if out:
                    logger.info(f"web_search via SearXNG {{engine or 'aggregate'}}: {query} -> {len(out)}")
                    return out
    except Exception as e:
        logger.debug(f"SearXNG({engine or 'agg'}) not available {e}")
    return None

async def _try_ddgs(query: str, count: int) -> List[Dict] | None:
    try:
        from duckduckgo_search import DDGS
        def _sync():
            with DDGS() as ddgs:
                res = list(ddgs.text(query, max_results=count, backend="lite"))
                # fallback to html if lite fails
                if not res:
                    res = list(ddgs.text(query, max_results=count))
                return res
        res = await asyncio.to_thread(_sync)
        if res:
            out = []
            for r in res[:count]:
                out.append({
                    "title": r.get("title",""),
                    "url": r.get("href") or r.get("url",""),
                    "content": r.get("body") or r.get("content",""),
                })
            logger.info(f"web_search via DDGS: {query} -> {len(out)}")
            return out
    except Exception as e:
        logger.debug(f"DDGS failed {e}")
    return None

async def _fetch_full_content(results: List[Dict], max_pages: int = 1, max_chars: int = 800) -> None:
    """Fetch full page text for top result only (fast, honest)."""
    if not results:
        return
    import asyncio
    targets = [r for r in results[:max_pages] if len(r.get("content","")) < 300]
    if not targets:
        return
    async def _fetch_one(r: Dict):
        url = r.get("url","")
        if not url or not url.startswith("http"):
            return
        try:
            async with httpx.AsyncClient(timeout=4.0, follow_redirects=True, headers={"User-Agent":"Mozilla/5.0"}) as client:
                resp = await client.get(url)
                if resp.status_code != 200 or len(resp.text) < 500:
                    return
                from lxml import html as lxh
                tree = lxh.fromstring(resp.text)
                # Remove scripts/styles/nav
                for el in tree.xpath('//script|//style|//nav|//header|//footer|//aside'):
                    el.getparent().remove(el) if el.getparent() is not None else None
                # Prefer article/main, fallback to all <p>
                paras = tree.xpath('//article//p//text() | //main//p//text()')
                if len(paras) < 3:
                    paras = tree.xpath('//div[contains(@class,"content")]//p//text()')
                if len(paras) < 3:
                    paras = tree.xpath('//p//text()')
                text = " ".join(p.strip() for p in paras if p.strip())
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 300:
                    # Append full text to existing snippet, keep snippet first
                    extra = text[:max_chars]
                    if extra not in r.get("content",""):
                        r["content"] = r.get("content","") + "\n\nFull page: " + extra
        except Exception:
            pass
    try:
        await asyncio.gather(*[_fetch_one(r) for r in targets])
    except Exception:
        pass

async def _try_lite_scrape(query: str, count: int) -> List[Dict] | None:
    """Scrape lite.duckduckgo.com as last network attempt (no API key)"""
    try:
        import httpx
        from lxml import html
        async with httpx.AsyncClient(timeout=5.0, headers={"User-Agent":"Mozilla/5.0"}) as client:
            resp = await client.get("https://lite.duckduckgo.com/lite/", params={"q": query})
            if resp.status_code != 200:
                return None
            tree = html.fromstring(resp.text)
            out = []
            # lite DDG structure: table with rows
            for row in tree.xpath('//table//tr[td]')[:count*2]:
                title_el = row.xpath('.//a[contains(@href,"://")]')
                snippet_el = row.xpath('.//td[@class="result-snippet"]//text()')
                if not title_el:
                    continue
                title = title_el[0].text_content().strip()[:120]
                url = title_el[0].get("href","")
                snippet = " ".join(snippet_el).strip()[:300] if snippet_el else ""
                if title and url:
                    out.append({"title": title, "url": url, "content": snippet})
                if len(out) >= count:
                    break
            if out:
                logger.info(f"web_search via lite scrape: {query} -> {len(out)}")
                return out
    except Exception as e:
        logger.debug(f"lite scrape failed {e}")
    return None

async def _try_bing(query: str, count: int) -> List[Dict] | None:
    """Scrape Bing as honest fallback when DDG/lite are CAPTCHA-blocked (no API key)."""
    q = query.strip()
    if q.lower().startswith("!bing "):
        q = q[6:].strip()
    try:
        import httpx
        from lxml import html as lhtml
        async with httpx.AsyncClient(timeout=5.0, headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}, follow_redirects=True) as client:
            resp = await client.get("https://www.bing.com/search", params={"q": q, "form":"QBRE"})
            if resp.status_code != 200:
                return None
            tree = lhtml.fromstring(resp.text)
            out = []
            for li in tree.xpath('//li[contains(@class,"b_algo")]')[:count*2]:
                a = li.xpath('.//h2/a')
                if not a:
                    continue
                title = a[0].text_content().strip()[:140]
                url = a[0].get("href","")
                snippet = " ".join(li.xpath('.//p//text()')).strip()[:320]
                if not snippet:
                    snippet = " ".join(li.xpath('.//div[contains(@class,"b_caption")]//p//text()')).strip()[:320]
                if title and url and url.startswith("http"):
                    out.append({"title": title, "url": url, "content": snippet})
                if len(out) >= count:
                    break
            if out:
                logger.info(f"web_search via bing scrape: {q} -> {len(out)}")
                return out
    except Exception as e:
        logger.debug(f"bing scrape failed {e}")
    return None

_STOP_ZH = {"\u6700\u65b0","\u4eca\u5929","\u4eca\u5e74","\u73fe\u5728","\u76ee\u524d","\u4e00\u4e0b","\u7684","\u4e86","\u662f","\u5728","\u8207","\u548c","\u53ca","\u6709","\u55ce","\u5462","\u554a"}
_STOP_EN = {"the","is","are","what","who","when","where","how","why","a","an","in","on","for","to","of","and","or","latest","today","current","search"}

def _tokenize_query(q: str) -> List[str]:
    """CJK-aware query tokenization for relevance scoring. A naive regex
    (`[\\u4e00-\\u9fff]+`) treats an entire unsegmented Chinese phrase as ONE
    token (e.g. '\u6700\u65b0\u79d1\u6280\u65b0\u805e' as a single 6-char blob), which then almost never
    appears verbatim in a result's title/content/url \u2014 so every Chinese query
    scored ~0 regardless of how relevant the results actually were, silently
    defeating the fallback/reformulation logic this score is supposed to
    drive. Use jieba to split CJK runs into real words (\u79d1\u6280/\u65b0\u805e/...) so
    matching works the same way it does for space-delimited English."""
    import jieba
    tokens = []
    for run in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", q.lower()):
        if re.match(r"[\u4e00-\u9fff]", run):
            tokens.extend(w.strip() for w in jieba.cut(run) if len(w.strip()) >= 2 and w.strip() not in _STOP_ZH)
        elif len(run) >= 3 and run not in _STOP_EN:
            tokens.append(run)
    return tokens

def _relevance_score(query: str, results: List[Dict]) -> float:
    """Score 0..1 how relevant results are to query. Low = bad search, should fallback.

    A query that yields NO scorable tokens (pure digits, emoji, a 1-2 char CJK
    bigram that the >=2-char filter drops, stopwords-only) is scored 1.0 = neutral,
    NOT 0. "No tokens" means "I cannot judge this", which is not evidence of a bad
    search — scoring it 0 made every such query permanently "irrelevant", so its
    cache entry was always rejected and the whole multi-second fallback chain re-ran
    on every single request with no possibility of ever improving the score."""
    if not results:
        return 0
    tokens = _tokenize_query(query)
    if not tokens:
        return 1.0
    # Require majority of tokens to appear for relevance (not just any one)
    hits = 0
    for r in results[:3]:
        text = (r.get("title","")+" "+r.get("content","")+" "+r.get("url","")).lower()
        matched = sum(1 for tok in tokens if tok in text)
        # Need at least half the tokens, or all if only 1-2 tokens
        needed = max(1, (len(tokens)+1)//2)  # ceil(len/2)
        if len(tokens) == 1:
            needed = 1
        elif len(tokens) == 2:
            needed = 2  # for 2 tokens like "weather paris", need both
        if matched >= needed:
            hits += 1
    score = hits / min(3, len(results))
    return score

EMBED_URL = os.getenv("EMBED_API_BASE", "http://127.0.0.1:11434/v1/embeddings")
if not EMBED_URL.endswith("/embeddings"):
    EMBED_URL = EMBED_URL.rstrip("/") + "/embeddings"
EMBED_MODEL = os.getenv("EMBED_MODEL", "granite-embedding")

async def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Batch embed via granite-97m Q8 GGUF on :11434 (115MB, 384d, zh-TW+en, CUDA)."""
    if not texts:
        return []
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts})
            if resp.status_code == 200:
                data = resp.json()
                items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                return [it["embedding"] for it in items if "embedding" in it]
    except Exception as e:
        logger.debug(f"granite embed failed {e}")
    return []

def _cosine(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x*y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x*x for x in a)) or 1e-9
    nb = math.sqrt(sum(y*y for y in b)) or 1e-9
    return dot / (na * nb)

async def _rerank_with_embeddings(query: str, results: List[Dict]) -> tuple[List[Dict], float]:
    """Semantic rerank via granite-97m-multilingual (384d, zh-TW+en). Returns (reranked_results, semantic_score 0..1)."""
    if not results or len(results) < 2:
        return results, _relevance_score(query, results)
    try:
        texts = [f"{r.get('title','')} {r.get('content','')[:400]}" for r in results]
        embs = await _embed_batch([query] + texts)
        if len(embs) != len(texts) + 1:
            return results, _relevance_score(query, results)
        q_emb = embs[0]
        scored = []
        for r, e in zip(results, embs[1:], strict=False):
            s = _cosine(q_emb, e)
            # map cosine -1..1 to 0..1, granite typical 0.4..0.9 for relevant
            norm = (s + 1) / 2
            scored.append((norm, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [r for _, r in scored]
        # semantic score = mean top-3 cosine norm
        top = [s for s, _ in scored[:3]]
        sem_score = sum(top) / len(top) if top else 0
        # Convert to 0..1 relevance comparable to keyword score: threshold ~0.65 is good
        # Keep as 0..1 for decision logic; caller can blend
        return reranked, sem_score
    except Exception as e:
        logger.debug(f"embedding rerank failed {e}")
        return results, _relevance_score(query, results)

def _mock_search(query: str, count: int) -> List[Dict]:
    """Generic mock only for truly offline case — no curated cheating"""
    return [
        {"title": f"Search results for '{query}' — mock result {i+1}", "url": f"https://example.com/search?q={query.replace(' ','+')}&i={i}", "content": f"This is a mock snippet for query '{query}'. The assistant is in mock mode without network, but tool calling pipeline is working. Result {i+1} contains synthesized information about {query}."}
        for i in range(min(count,3))
    ]

def _sanitize_query(q: str) -> str:
    """Strip language-hint and other system-prompt leakage from tool queries.

    Observed: `搜尋最新的科技新聞（請一律使用繁體中文…）` — the LANG_HINT that was appended
    to the user turn got copied verbatim into web_search's `query` arg, producing a
    literal parenthetical search that returns `no_results`. This is defense-in-depth:
    the hint is no longer appended on the agent path, but any hint that still slips
    through is removed here so the search itself stays honest."""
    if not q:
        return q
    # Remove the Traditional-Chinese language hint that used to be appended to user text
    if "（請一律使用繁體中文" in q:
        q = q.split("（請一律使用繁體中文")[0].strip()
    if "(請一律使用" in q:
        q = q.split("(請一律使用")[0].strip()
    # Also strip any trailing parenthetical that is purely the language instruction
    q = re.sub(r"[（(]請一律.*?[）)]\s*$", "", q).strip()
    # Remove quoted instruction leakage like 'Never do more than 2 tool calls...'
    if len(q) > 60 and "Never do more than" in q:
        q = q.split("Never do more than")[0].strip()
    return q.strip(" \t\n，。、；：")


async def web_search(query: str, count: int = 5) -> Dict:
    """
    Tool function: web_search
    Returns {"query": str, "results": List[Dict], "source": str, "latency_ms": int}
    """
    # Robustness: model sometimes sends JSON string '{"query": "..."}'
    _raw = query.strip() if isinstance(query, str) else str(query).strip()
    if _raw.startswith('{'):
        try:
            import json as _js
            d = _js.loads(_raw)
            if isinstance(d, dict) and 'query' in d:
                _raw = str(d['query']).strip()
        except Exception:
            _raw = repair_truncated_json_query(_raw)
    query = _sanitize_query(_raw.strip())
    if not query:
        return {"query": query, "results": [], "source": "none", "latency_ms": 0}
    ck = _cache_key(query, count)
    cached = _cache_get(ck)
    if cached is not None:
        # Check relevance even for cache - if cached is bad, don't use it
        if _relevance_score(query, cached.get("results", [])) < 0.34 and not cached.get("source", "").startswith("mock"):
            logger.info(f"web_search CACHE hit but irrelevant for '{query}' (score {_relevance_score(query, cached.get('results',[])):.2f}), ignoring cache")
        else:
            logger.info(f"web_search CACHE hit: {query}")
            return {**cached, "latency_ms": 2, "cached": True}

    t0 = time.time()
    def _score(rs):
        return _relevance_score(query, rs) if rs else 0.0
    # Helper to enrich top results with full page text (for more informative answers)
    async def _maybe_enrich(results: List[Dict] | None):
        if results and len(results) >= 1:
            # Enrich when snippet is thin or query asks for news/details
            thin = any(len(r.get("content","")) < 300 for r in results[:2])
            is_detailed = bool(re.search(r"(news|headlines|details|explain|summary|full|article|新闻|头条)", query.lower()))
            if thin or is_detailed:
                await _fetch_full_content(results, max_pages=2, max_chars=1200)
    # 0) direct weather source (wttr.in) — zh+en weather queries get real forecast numbers,
    #    avoiding generic bing pages (baike/attractions) that make the LLM refuse to answer.
    wttr = await _wttr_weather(query)
    if wttr:
        results = await _try_searxng(query, count)
        if results:
            results.insert(0, wttr)
        else:
            results = [wttr]
        # Honest label of what is actually in the payload (was a dead `src_extra`/`source_was` pair).
        source = "wttr.in+searxng" if len(results) > 1 else "wttr.in"
        best_score = 1.0
        payload = {"query": query, "results": results[:count], "source": source,
                   "latency_ms": int((time.time()-t0)*1000), "cached": False}
        _cache_put(ck, payload)
        logger.info(f"web_search '{query}' source={source} (wttr+searxng) {len(results)} results")
        return payload
    # 1) aggregate SearXNG (auto language)
    results = await _try_searxng(query, count)
    source = "searxng"
    best_score = _score(results)
    logger.info(f"SearXNG relevance for '{query}' = {best_score:.2f}")
    # 1b) embedding rerank when keyword score is uncertain (0.34-0.65) — semantic helps paraphrases like "big news days"
    if results and 0.34 <= best_score < 0.65:
        try:
            reranked, sem_score = await _rerank_with_embeddings(query, results)
            if sem_score > 0.65 and reranked is not results:
                logger.info(f"  -> embedding rerank improved {best_score:.2f} -> {sem_score:.2f}")
                results = reranked
                best_score = max(best_score, sem_score * 0.85)  # blend, keep honest
                source = f"{source}+emb"
        except Exception as e:
            logger.debug(f"embedding rerank skip {e}")
    # 2) if low relevance -> crafted entity-first queries (honest, generic, max 1 candidate for speed)
    if results is None or best_score < 0.34:
        done = False
        for alt_q in _entity_first_query(query)[:1]:
            for qq, tag in (("!bing " + alt_q, "bing"), (alt_q, "agg")):
                r2 = await _try_searxng(qq, count, engine=None)
                s2 = _score(r2)
                if r2 and s2 > best_score:
                    results, source, best_score = r2, f"searxng:{tag}", s2
                    logger.info(f"  -> better via {tag} q='{qq[:40]}' (score {s2:.2f})")
                if best_score >= 0.34:
                    done = True
                    break
            if done:
                break
    # 3) network fallbacks — race concurrently instead of walking them one at a
    # time. Each stage carries its own several-second httpx timeout, and
    # empirically most of them fail outright for a given query before
    # whichever stage actually succeeds is reached, so walking them serially
    # wastes seconds waiting on dead ends (measured ~3s for a typical zh-TW
    # query). DDGS is a US/English-oriented backend that never wins for
    # Chinese queries in practice, so skip its timeout there entirely.
    if best_score < 0.34:
        candidates = [] if _is_chinese(query) else [("duckduckgo", _try_ddgs(query, count))]
        candidates += [("lite_scrape", _try_lite_scrape(query, count)), ("bing_scrape", _try_bing(query, count))]
        gathered = await asyncio.gather(*(c[1] for c in candidates), return_exceptions=True)
        for (name, _), r in zip(candidates, gathered, strict=False):
            if isinstance(r, Exception) or not r:
                continue
            s = _score(r)
            if s > best_score:
                results, source, best_score = r, name, s
    if not results:
        latency = int((time.time()-t0)*1000)
        payload = {"query": query, "results": [], "source": "no_results", "latency_ms": latency, "cached": False}
        _cache_put(ck, payload)
        logger.info(f"web_search '{query}' source=no_results 0 results in {latency}ms — no mock")
        return payload

    # Enrich top results with full page text for more informative answers (concurrent, capped)
    await _maybe_enrich(results)
    # Final embedding rerank after enrichment if still borderline
    if results and best_score < 0.5 and source not in ("wttr.in", "mock-offline"):
        try:
            reranked, sem_score = await _rerank_with_embeddings(query, results)
            if sem_score > 0.62:
                results = reranked
                source = f"{source}+emb-full"
        except Exception:
            pass

    latency = int((time.time()-t0)*1000)
    payload = {"query": query, "results": results[:count], "source": source, "latency_ms": latency, "cached": False}
    _cache_put(ck, payload)
    logger.info(f"web_search '{query}' source={source} {len(results)} results in {latency}ms")
    return payload

# Sync wrapper for non-async callers (the agent harnesses run their loop in a worker
# thread and call tools synchronously).
_SYNC_LOOP: asyncio.AbstractEventLoop | None = None
_SYNC_THREAD = None


def web_search_sync(query: str, count: int = 5) -> Dict:
    """Run web_search() from sync code.

    Uses one persistent background event loop instead of asyncio.run() per call:
    a fresh loop per tool call threw away the httpx connection pools (new TLS
    handshake to every engine on every search) and, if this function were ever
    called from a thread that already had a running loop, asyncio.run() would
    raise instead of searching.
    """
    global _SYNC_LOOP, _SYNC_THREAD
    if _SYNC_LOOP is None or _SYNC_THREAD is None or not _SYNC_THREAD.is_alive():
        _SYNC_LOOP = asyncio.new_event_loop()
        _SYNC_THREAD = threading.Thread(target=_SYNC_LOOP.run_forever, daemon=True, name="web-search-loop")
        _SYNC_THREAD.start()
    fut = asyncio.run_coroutine_threadsafe(web_search(query, count=count), _SYNC_LOOP)
    return fut.result(timeout=45)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "weather Paris"
    res = asyncio.run(web_search(q))
    print(f"Query: {res['query']} source={res['source']} latency={res['latency_ms']}ms")
    for r in res["results"]:
        print(f" - {r['title']} | {r['url']}\n   {r['content'][:120]}")
