"""
Tool definitions for MiniCPM5 — web_search via self-hosted SearXNG
"""
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for weather, news, facts, recent events, people, definitions, prices, stock, sports, current events. Always use for 'latest', 'current', 'today', 'who is', 'what is', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query, 3-8 words, specific"},
                    "count": {"type": "integer", "description": "number of results", "default": 5}
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM_PROMPT_WITH_TOOLS = """You are a helpful voice assistant. Keep replies concise, conversational, under 40 words. Speak naturally as if in a phone call.

TOOLS:
- web_search(query: string): Search the web for current information. Use it when the user asks about weather, news, recent events, people, facts, definitions, prices, or anything requiring current data.

TOOL CALLING:
If you need to search, output EXACTLY one JSON line and nothing else:
<tool>web_search{"query": "your query"}</tool>
The system will run the search and give you results, then you answer.

Otherwise answer directly. For voice, be brief and friendly.
"""

# Heuristic triggers — fast path without waiting for LLM to decide
TOOL_TRIGGERS = [
    "weather", "temperature", "forecast",
    "search", "look up", "find",
    "news", "latest", "recent", "today", "current", "now",
    "who is", "who's", "what is", "when is", "where is",
    "price", "stock", "score", "game", "match",
    "how much", "define", "definition",
    "openbmb", "minicpm", "searxng", "primetts",
    "python 3.14", "ai", "gpt", "huggingface"
]

def should_search_heuristic(prompt: str) -> tuple[bool, str]:
    """Fast heuristic: returns (should_search, suggested_query)"""
    pl = prompt.lower()
    for trig in TOOL_TRIGGERS:
        if trig in pl:
            # extract a query: use last 8 words or full prompt trimmed
            # For demo, just use prompt as query (trimmed)
            q = prompt.strip()
            # clean
            if len(q) > 80:
                # take last sentence
                q = q.split(".")[-1].strip() or q[:80]
            if len(q) > 60:
                q = " ".join(q.split()[-8:])
            return True, q
    return False, ""
