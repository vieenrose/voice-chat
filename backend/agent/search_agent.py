"""web_search as a sub-agent rather than a single tool call.

A one-shot search is only as good as the query the model happened to write, and
that turned out to be the whole failure mode: 「今天最重要的三條新聞」 retrieved a
Wikinews front page from 2025-10-01 and scored 0.00 relevance, while
「台灣 今日 頭條新聞」 returned that morning's headlines. The model then honestly
reported it could not find the news.

So searching gets its own bounded loop, with its own message list. Two properties
are deliberate:

**Retries are triggered by the score, not by an LLM opinion.** ``web_search``
already computes a relevance score and, until now, threw it away. Asking the model
"are these results good?" costs a prose generation -- measured at 4.1-7.6 s on this
box, against 0.6 s for a step that only emits a tool call. The objective score is
free and does not hallucinate, so it decides; the model is asked for one thing
only, a better query.

**Its context stays separate.** Failed queries and discarded result dumps never
enter the voice turn's message list, so the spoken answer is not built on top of
pages of search junk.

No framework: this is ``/v1/chat/completions`` with one tool, the same endpoint and
the same model as the parent turn.
"""

from __future__ import annotations

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

# Retry only when the results share essentially nothing with the query. The score
# is keyword overlap, and news headlines legitimately score low -- measured, the
# stale Wikinews page scored 0.00 while 台灣 今日 頭條新聞, which returned that
# morning's real headlines, scored 0.33. Reusing web_search's internal 0.34
# threshold therefore retried a *good* query and cost 15 s. 0.2 separates "no
# overlap at all" from "low overlap but on topic".
GOOD_ENOUGH = float(os.getenv("SEARCH_AGENT_MIN_RELEVANCE", "0.2"))

# Default 1: the retry is OFF unless asked for. Measured over three spoken news
# turns, enabling it moved median first-audio from 4.9 s to 8.6 s (3.9/4.9/6.5 ->
# 7.4/8.6/11.1) and produced no better answers -- once the tool schema let the
# model declare `recency` and gave it query guidance, the first search was
# already good, and the retry fired on 2 of 3 turns anyway because the relevance
# score reads low on news. Set SEARCH_AGENT_MAX_SEARCHES=2 to turn it on for a
# corpus where first searches genuinely fail.
MAX_SEARCHES = int(os.getenv("SEARCH_AGENT_MAX_SEARCHES", "1"))

_REWRITE_SYSTEM = (
    "You rewrite failed web-search queries. Reply with a single get_query tool call "
    "and nothing else. Keep it 3-8 keywords, drop question words, and name the place "
    "and the recency when the question is about now."
)

# Asking for a *query* rather than an answer keeps this to one short tool call.
_REWRITE_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_query",
        "description": "Supply a better search query for the user's question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "3-8 keywords"},
                "recency": {"type": "string", "enum": ["any", "day", "week"]},
            },
            "required": ["query"],
        },
    },
}]


def _better_query(question: str, tried: list[tuple[str, float]], *, api_base: str,
                  model: str, api_key: str, generate_cfg: dict) -> tuple[str, str] | None:
    """Ask the model for one improved query. Returns None if it does not comply."""
    attempts = "\n".join(f"- {q!r} scored {s:.2f}" for q, s in tried)
    cfg = {k: v for k, v in (generate_cfg or {}).items() if k not in ("model", "stream")}
    body = {
        "model": model, **cfg, "tools": _REWRITE_TOOL, "tool_choice": "auto",
        "messages": [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nQueries already tried:\n"
                                        f"{attempts}\nGive a better query."},
        ],
    }
    try:
        r = requests.post(f"{api_base.rstrip('/')}/chat/completions", json=body,
                          headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        r.raise_for_status()
        calls = (r.json()["choices"][0]["message"].get("tool_calls") or [])
        for c in calls:
            args = json.loads(c["function"]["arguments"] or "{}")
            q = str(args.get("query", "")).strip()
            if q:
                rec = args.get("recency", "any")
                return q, (rec if rec in ("any", "day", "week") else "any")
    except Exception:
        logger.exception("query rewrite failed; keeping the first result set")
    return None


def search(question: str, query: str, recency: str = "any", *, count: int = 5,
           api_base: str, model: str, api_key: str = "none",
           generate_cfg: dict | None = None) -> dict:
    """Run the search, and retry once with a better query if the first scored badly.

    `query`/`recency` are the parent turn's own choice, so a good first query costs
    exactly what it did before: one search, no extra generation.
    """
    from tools.web_search import web_search_sync

    best = web_search_sync(query, count=count, recency=recency)
    tried = [(query, float(best.get("relevance", 0.0)))]
    if tried[0][1] >= GOOD_ENOUGH or MAX_SEARCHES < 2:
        return best

    logger.info("search agent: %r scored %.2f; retrying", query, tried[0][1])
    for _ in range(MAX_SEARCHES - 1):
        nxt = _better_query(question, tried, api_base=api_base, model=model,
                            api_key=api_key, generate_cfg=generate_cfg or {})
        if not nxt or nxt[0] == query:
            break
        q2, rec2 = nxt
        res = web_search_sync(q2, count=count, recency=rec2)
        score = float(res.get("relevance", 0.0))
        tried.append((q2, score))
        if score > float(best.get("relevance", 0.0)):
            best = res
        if score >= GOOD_ENOUGH:
            break
    logger.info("search agent: tried %s -> kept %r (%.2f)",
                [q for q, _ in tried], best.get("query"), float(best.get("relevance", 0.0)))
    return best
