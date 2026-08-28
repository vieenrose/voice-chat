"""
Ling 3.0 Tiny MXFP4 via llama-cpp-python (JamePeng fork) — drop-in replacement for ling_streaming.py
Proves llama-cpp-python is easier to integrate: no HTTP server, direct Python call, same interface.
Falls back to HTTP if llama_cpp not installed, so you can A/B test latency.
"""
import asyncio
import time
import json
from typing import AsyncGenerator, List, Dict
from loguru import logger

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current info. Use for weather, news, facts, recent events.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "count": {"type": "integer", "default": 5}}, "required": ["query"]}
        }
    }
]
SYSTEM_PROMPT = "You are a helpful voice assistant with web search. Keep replies concise, conversational, under 80 words, speak naturally for voice chat. For ANY question about current events, news, weather, real-time info, you MUST use web_search tool. Always respond in the user's language."

TOOL_TRIGGERS = ["weather","temperature","forecast","search","news","latest","today","current","who is","what is","when","where","price","stock","define","python","ai","gpt","ling","新闻","事件","台湾","台灣","今天","最新","台北"]
TOOL_EXCLUDE_PHRASES = ["my name", "my age", "my birthday", "remember it", "recall", "我的名字"]
def should_search_heuristic(prompt: str):
    pl = prompt.lower()
    for excl in TOOL_EXCLUDE_PHRASES:
        if excl in pl: return False,""
    for trig in TOOL_TRIGGERS:
        if trig in pl:
            q = prompt.strip()[:80]
            if len(q)>60: q=" ".join(q.split()[-8:])
            return True, q
    return False,""

class LingCpp:
    """Same interface as LingStreaming but via llama_cpp.Llama directly (no HTTP)."""
    def __init__(self, model_path: str = "/tmp/Ling/Ling-3.0-tiny-MXFP4_MOE.gguf", n_ctx: int = 8192, n_gpu_layers: int = 99, verbose: bool = False, **kwargs):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.backend = "ling-cpp-python"
        self.mock = False
        self.llm = None
        try:
            from llama_cpp import Llama
            logger.info(f"LingCpp loading {model_path} n_gpu_layers={n_gpu_layers}...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                chat_format="chatml",  # Ling uses chatml-like <role>
                n_threads=8,
            )
            logger.info(f"LingCpp ready ✓ {model_path} GPU layers {n_gpu_layers}")
        except Exception as e:
            logger.warning(f"LingCpp not available {e}, will use HTTP fallback (llama-server). Import error means pip install not done yet.")
            self.llm = None
            self.backend = "ling-cpp-missing-fallback-http"
            # lazy HTTP fallback
            from .ling_streaming import LingStreaming
            self.fallback = LingStreaming()

    async def _stream_cpp(self, messages: List[Dict], tools=None, max_tokens=256):
        if self.llm is None:
            # fallback to HTTP
            async for ev in self.fallback._chat_stream(messages, tools=tools, max_tokens=max_tokens):
                yield ev
            return
        # Run blocking llama_cpp in thread
        def _gen():
            # llama_cpp create_chat_completion with stream=True
            kwargs = dict(messages=messages, max_tokens=max_tokens, temperature=0.7, top_p=0.9, stream=True)
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return self.llm.create_chat_completion(**kwargs)

        # Need to handle sync generator in async context
        loop = asyncio.get_event_loop()
        gen = await loop.run_in_executor(None, _gen)
        for chunk in gen:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    yield {"type": "tool_call_delta", "delta": tc}
            token = delta.get("content")
            if token is not None:
                yield {"type": "token", "token": token}
            if choice.get("finish_reason"):
                yield {"type": "finish", "reason": choice["finish_reason"]}

    async def generate_chat(self, history: List[Dict], prompt: str = None, max_new_tokens: int = 256) -> AsyncGenerator[dict, None]:
        messages = []
        if not history or history[0].get("role") != "system":
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
            if history: messages.extend(history)
        else:
            messages = list(history)
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
        if self.llm is None:
            async for ev in self.fallback.generate_chat(history, prompt, max_new_tokens):
                yield ev
            return
        text_so_far=""; t0=time.time(); first=True
        try:
            async for ev in self._stream_cpp(messages, tools=None, max_tokens=max_new_tokens):
                if ev["type"]=="token":
                    token=ev["token"]; text_so_far+=token
                    latency=int((time.time()-t0)*1000) if first else 20; first=False
                    yield {"type":"llm_token","token":token,"text_so_far":text_so_far,"latency_ms":latency}
                    await asyncio.sleep(0)
            yield {"type":"llm_done","text":text_so_far}
        except Exception as e:
            logger.exception(f"LingCpp generate_chat failed {e}")
            yield {"type":"llm_done","text":text_so_far}

    async def generate_chat_with_tools(self, history: List[Dict], prompt: str, max_new_tokens: int = 256) -> AsyncGenerator[dict, None]:
        try:
            from tools.web_search import web_search, format_results
        except:
            web_search=None
        should, query = should_search_heuristic(prompt) if "web search results" not in prompt.lower() else (False,"")
        if should and web_search is not None:
            logger.info(f"[LingCpp Tool] heuristic web_search '{query}'")
            yield {"type":"tool_call","name":"web_search","arguments":{"query":query},"query":query}
            t_tool=time.time()
            try:
                search_res = await web_search(query, count=5)
                formatted = format_results(search_res["results"])
                yield {"type":"tool_result","name":"web_search","result":search_res,"formatted":formatted,"latency_ms":int((time.time()-t_tool)*1000)}
                # Build augmented messages with tool observation
                augmented = list(history) if history else []
                if not augmented or augmented[0].get("role")!="system":
                    augmented=[{"role":"system","content":SYSTEM_PROMPT}]+augmented
                augmented.append({"role":"user","content":prompt})
                augmented.append({"role":"assistant","tool_calls":[{"id":"call_0","type":"function","function":{"name":"web_search","arguments":json.dumps({"query":query})}}]})
                augmented.append({"role":"tool","tool_call_id":"call_0","content":formatted})
                # Generate final answer
                text_so_far=""; t0=time.time(); first=True
                async for ev in self._stream_cpp(augmented, tools=None, max_tokens=max_new_tokens):
                    if ev["type"]=="token":
                        token=ev["token"]; text_so_far+=token
                        latency=int((time.time()-t0)*1000) if first else 20; first=False
                        yield {"type":"llm_token","token":token,"text_so_far":text_so_far,"latency_ms":latency}
                yield {"type":"llm_done","text":text_so_far}
                return
            except Exception as e:
                logger.exception(f"LingCpp tool failed {e}")
                yield {"type":"tool_result","name":"web_search","error":str(e)}
        # No tool, normal chat
        async for ev in self.generate_chat(history, prompt, max_new_tokens):
            yield ev

    async def generate_stream(self, prompt: str, max_new_tokens: int = 256):
        async for ev in self.generate_chat([], prompt, max_new_tokens):
            yield ev
    async def generate_with_tools(self, prompt: str, max_new_tokens: int = 256):
        async for ev in self.generate_chat_with_tools([], prompt, max_new_tokens):
            yield ev
