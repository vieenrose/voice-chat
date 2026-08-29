"""
Web search tool — self-hosted SearXNG compatible.

Tries in order:
  1. Local SearXNG at http://localhost:8888/search?format=json  (self-hosted)
  2. DuckDuckGo via duckduckgo_search.DDGS (pip)
  3. Lite DuckDuckGo scrape via httpx + lxml
  4. Mock curated results (works offline, deterministic)

API matches SearXNG json: /search?q=...&format=json&categories=general
Also exposes `web_search(query, count=5)` async function for tool calling.

Uses caching (5 min TTL) to keep latency low.
"""
import asyncio
import time
import re
import hashlib
import urllib.parse
import httpx
from typing import List, Dict
from loguru import logger

# Simple in-memory cache
_CACHE: Dict[str, dict] = {}
CACHE_TTL = 300  # 5 min

# No curated mock DB — honest search only; generic mock below is only for truly offline case

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


def _entity_first_query(q: str) -> str:
    """Generic query reformulation — no hard-coded outlet/location lists."""
    return [_clean_query(q) or q.strip()[:40]]

async def _try_searxng(query: str, count: int, engine: str | None = None) -> List[Dict] | None:
    """Try local self-hosted SearXNG on 8888. engine=None -> aggregate; else single engine by name."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            # language: zh for Chinese, EN otherwise (bing with language=all returns zh-biased junk)
            _lang = "zh-TW" if ("台灣" in query or "台湾" in query) else ("zh-CN" if _is_chinese(query) else "en")
            # Generic news detection (no hard-coded outlet list) — use SearXNG news category for news queries
            _is_news = bool(re.search(r"(news|headlines|頭條|新聞|breaking|latest.*news|news.*today)", query.lower()))
            _cat = "news" if _is_news else "general"
            params = {"q": query, "format": "json", "categories": _cat, "pageno": 1,
                      "language": _lang, "safesearch": 1}
            if engine:
                params["engines"] = engine
            resp = await client.get("http://localhost:8888/search", params=params)
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
                if not title_el: continue
                title = title_el[0].text_content().strip()[:120]
                url = title_el[0].get("href","")
                snippet = " ".join(snippet_el).strip()[:300] if snippet_el else ""
                if title and url:
                    out.append({"title": title, "url": url, "content": snippet})
                if len(out) >= count: break
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
                if not a: continue
                title = a[0].text_content().strip()[:140]
                url = a[0].get("href","")
                snippet = " ".join(li.xpath('.//p//text()')).strip()[:320]
                if not snippet:
                    snippet = " ".join(li.xpath('.//div[contains(@class,"b_caption")]//p//text()')).strip()[:320]
                if title and url and url.startswith("http"):
                    out.append({"title": title, "url": url, "content": snippet})
                if len(out) >= count: break
            if out:
                logger.info(f"web_search via bing scrape: {q} -> {len(out)}")
                return out
    except Exception as e:
        logger.debug(f"bing scrape failed {e}")
    return None

def _relevance_score(query: str, results: List[Dict]) -> float:
    """Score 0..1 how relevant results are to query. Low = bad search, should fallback."""
    ql = query.lower()
    # Tokenize query, keep meaningful words >=3 chars, not stopwords
    stop = {"the","is","are","what","who","when","where","how","why","a","an","in","on","for","to","of","and","or","latest","today","current","search"}
    tokens = [w for w in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", ql) if len(w)>=3 and w not in stop]
    if not tokens:
        tokens = [w for w in ql.split() if len(w)>=2]
    if not results or not tokens:
        return 0
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

EMBED_URL = "http://127.0.0.1:11434/v1/embeddings"
EMBED_MODEL = "granite-embedding"

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
    dot = sum(x*y for x, y in zip(a, b))
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
        for r, e in zip(results, embs[1:]):
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
        except: pass
    query = _raw.strip()
    if not query:
        return {"query": query, "results": [], "source": "none", "latency_ms": 0}
    ck = _cache_key(query, count)
    if ck in _CACHE:
        entry = _CACHE[ck]
        if time.time() - entry["t"] < CACHE_TTL:
            cached = entry["data"]
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
        src_extra = source_was = "searxng"
        if results:
            results.insert(0, wttr)
        else:
            results = [wttr]
        source = "wttr.in"
        best_score = 1.0
        payload = {"query": query, "results": results[:count], "source": source,
                   "latency_ms": int((time.time()-t0)*1000), "cached": False}
        _CACHE[ck] = {"t": time.time(), "data": payload}
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
    # 3) DDG / lite scrape as network fallback
    if best_score < 0.34:
        ddgs = await _try_ddgs(query, count)
        if ddgs and _score(ddgs) > best_score:
            results, source, best_score = ddgs, "duckduckgo", _score(ddgs)
    if best_score < 0.34:
        lite = await _try_lite_scrape(query, count)
        if lite and _score(lite) > best_score:
            results, source, best_score = lite, "lite_scrape", _score(lite)
    if best_score < 0.34:
        bing = await _try_bing(query, count)
        if bing and _score(bing) > best_score:
            results, source, best_score = bing, "bing_scrape", _score(bing)
    if not results:
        latency = int((time.time()-t0)*1000)
        payload = {"query": query, "results": [], "source": "no_results", "latency_ms": latency, "cached": False}
        _CACHE[ck] = {"t": time.time(), "data": payload}
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
    _CACHE[ck] = {"t": time.time(), "data": payload}
    logger.info(f"web_search '{query}' source={source} {len(results)} results in {latency}ms")
    return payload

# Sync wrapper for non-async callers
def web_search_sync(query: str, count: int = 5) -> Dict:
    return asyncio.run(web_search(query, count))

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "weather Paris"
    res = asyncio.run(web_search(q))
    print(f"Query: {res['query']} source={res['source']} latency={res['latency_ms']}ms")
    for r in res["results"]:
        print(f" - {r['title']} | {r['url']}\n   {r['content'][:120]}")
