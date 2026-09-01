"""
MiniCPM5 streaming LLM wrapper.

Spec: "llm: minicpm5" → openbmb/MiniCPM5-1B
Streams tokens via TextIteratorStreamer (HF) with low TTFT.
Falls back to mock streaming if model not available / CUDA OOM.
"""
import asyncio
import time
import threading
from typing import AsyncGenerator
from loguru import logger

SYSTEM_PROMPT = "You are a helpful voice assistant. Keep replies concise, conversational, under 40 words. Speak naturally as if in a phone call."

# Import tool definitions
try:
    from .minicpm_tool import SYSTEM_PROMPT_WITH_TOOLS, TOOL_DEFS, should_search_heuristic
except ImportError:
    from minicpm_tool import SYSTEM_PROMPT_WITH_TOOLS, TOOL_DEFS, should_search_heuristic

TOOL_CALL_RE = __import__('re').compile(r'<tool>\s*(\w+)\s*(\{.*?\})\s*</tool>', __import__('re').DOTALL)
JSON_TOOL_RE = __import__('re').compile(r'\{\s*"tool"\s*:\s*"web_search".*?\}', __import__('re').DOTALL)

class MiniCPMStreaming:
    def __init__(self, model_id: str = "openbmb/MiniCPM5-1B", device: str = "cuda", mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.mock = mock
        self.tokenizer = None
        self.model = None
        self.streamer = None
        if mock:
            logger.info("MiniCPM5: MOCK mode")
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info(f"Loading MiniCPM5 {model_id} on {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            # MiniCPM5 uses bf16 on GPU
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=dtype,
                device_map="auto" if device=="cuda" else None, low_cpu_mem_usage=True
            )
            if device != "cuda":
                self.model = self.model.to(device)
            self.model.eval()
            logger.info("MiniCPM5 loaded ✓")
        except Exception as e:
            logger.error(f"MiniCPM5 load failed {e}, fallback to mock")
            self.mock = True

    async def generate_stream(self, prompt: str, max_new_tokens: int = 128) -> AsyncGenerator[dict, None]:
        """
        Yields {"type": "llm_token", "token": str, "text_so_far": str, "latency_ms": int}
        First token latency measured from call start.
        """
        t0 = time.time()
        if self.mock:
            # mock streaming: simulate token by token
            mock_resp = "That's interesting! I'm your voice assistant powered by MiniCPM5. How can I help you today?"
            # For some prompts, vary response
            if "weather" in prompt.lower():
                mock_resp = "I don't have live weather, but I can chat! Tell me your city and I'll pretend it's sunny and 22 degrees."
            elif "hello" in prompt.lower() or "hi" in prompt.lower():
                mock_resp = "Hey there! Great to hear your voice. What would you like to talk about?"
            text_so_far = ""
            for tok in mock_resp.split(" "):
                await asyncio.sleep(0.018)  # 18ms per token ~55 tok/s (optimized for <800ms E2E)
                text_so_far += (" " if text_so_far else "") + tok
                yield {"type": "llm_token", "token": " " + tok if text_so_far else tok, "text_so_far": text_so_far, "latency_ms": int((time.time()-t0)*1000)}
                t0 = time.time()  # subsequent tokens latency
            yield {"type": "llm_done", "text": text_so_far}
            return

        # Real streaming via TextIteratorStreamer in thread
        try:
            import torch
            from transformers import TextIteratorStreamer

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            # MiniCPM uses chat template if available
            try:
                input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            inputs = self.tokenizer(input_text, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.cuda() for k,v in inputs.items()}

            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30)
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
            thread.start()

            text_so_far = ""
            first = True
            t_start = time.time()
            for token_text in streamer:
                # token_text may be multiple chars
                text_so_far += token_text
                latency = int((time.time() - t_start)*1000) if first else int(40) # estimate
                first = False
                yield {"type": "llm_token", "token": token_text, "text_so_far": text_so_far, "latency_ms": latency}
                # allow event loop
                await asyncio.sleep(0)

            yield {"type": "llm_done", "text": text_so_far}
        except Exception as e:
            logger.exception(f"MiniCPM generate error {e}")
            # fallback mock
            yield {"type": "llm_token", "token": " Sorry, I had an error.", "text_so_far": "Sorry, I had an error.", "latency_ms": 50}
            yield {"type": "llm_done", "text": "Sorry, I had an error."}

    def generate_sync(self, prompt: str) -> str:
        import asyncio
        async def _collect():
            out = ""
            async for chunk in self.generate_stream(prompt):
                if chunk["type"] == "llm_token":
                    out = chunk["text_so_far"]
            return out
        return asyncio.run(_collect())

    async def generate_with_tools(self, prompt: str, max_new_tokens: int = 128) -> AsyncGenerator[dict, None]:
        """
        Tool-aware streaming generation for MiniCPM5.
        Yields events: llm_token, tool_call, tool_result, llm_done
        Supports web_search via self-hosted SearXNG.
        Works in both mock and real mode.
        """
        import json
        # Lazy import to avoid circular deps
        try:
            from tools.web_search import web_search, format_results
        except ImportError:
            try:
                from ..tools.web_search import web_search, format_results
            except Exception:
                web_search = None
                def format_results(x):
                    # fallback shim: tools.web_search is absent, so there is nothing to
                    # format — repr() is the honest degradation (E731: a def, not a lambda)
                    return str(x)

        # Fast-path heuristic: check if prompt likely needs search and we haven't already included search results
        is_search_prompt = "web search results" not in prompt.lower() and "search results for" not in prompt.lower()
        should, suggested_query = should_search_heuristic(prompt) if is_search_prompt else (False, "")

        if should and web_search is not None:
            # Emit tool call immediately (low-latency, saves one LLM round-trip)
            query = suggested_query or prompt[:60]
            logger.info(f"[Tool] heuristic triggers web_search for prompt='{prompt[:60]}' -> query='{query}'")
            yield {"type": "tool_call", "name": "web_search", "arguments": {"query": query}, "query": query}
            t_tool = time.time()
            try:
                search_res = await web_search(query, count=5)
                tool_latency = int((time.time() - t_tool)*1000)
                formatted = format_results(search_res["results"])
                yield {"type": "tool_result", "name": "web_search", "result": search_res, "formatted": formatted, "latency_ms": tool_latency}
                # Now generate final answer with search context
                augmented = f"User asked: {prompt}\n\nWeb search results for '{query}' (via SearXNG {search_res['source']}, {tool_latency}ms):\n{formatted}\n\nAnswer concisely in a natural voice-chat style, cite sources briefly, under 50 words."
                # For mock, craft a deterministic answer that includes search info
                if self.mock:
                    # Build mock answer from top result
                    top = search_res["results"][0] if search_res["results"] else {"content": "no results"}
                    mock_answer = f"I searched for '{query}' via SearXNG ({search_res['source']}). Top result: {top['title']}. {top['content'][:120]} — that's from {top['url'][:40]}. "
                    if "weather" in query.lower():
                        mock_answer = f"I checked via SearXNG — {top['content'][:140]} Current Paris weather is nice for a chat!"
                    # Stream mock answer
                    t0 = time.time()
                    text_so_far = ""
                    for tok in mock_answer.split(" "):
                        await asyncio.sleep(0.018)
                        text_so_far += (" " if text_so_far else "") + tok
                        yield {"type": "llm_token", "token": " " + tok if text_so_far else tok, "text_so_far": text_so_far, "latency_ms": int((time.time()-t0)*1000)}
                        t0 = time.time()
                    yield {"type": "llm_done", "text": text_so_far}
                    return
                # Real model: stream with augmented prompt
                async for tok in self.generate_stream(augmented, max_new_tokens=max_new_tokens):
                    yield tok
                return
            except Exception as e:
                logger.exception(f"tool search failed {e}")
                yield {"type": "tool_result", "name": "web_search", "error": str(e), "latency_ms": int((time.time()-t_tool)*1000)}
                # fallback to normal generation
                async for tok in self.generate_stream(prompt, max_new_tokens=max_new_tokens):
                    yield tok
                return

        # No heuristic hit — try normal generation and detect tool calls in output
        # Buffer tokens to detect <tool> pattern
        buffered = ""
        saw_tool_call = False
        t0 = time.time()
        async for ev in self.generate_stream(prompt, max_new_tokens=max_new_tokens):
            if ev["type"] == "llm_token":
                buffered = ev["text_so_far"]
                # Check for tool call pattern in buffered text
                m = TOOL_CALL_RE.search(buffered)
                if m and not saw_tool_call and web_search is not None:
                    saw_tool_call = True
                    tool_name = m.group(1)
                    try:
                        args = json.loads(m.group(2))
                        query = args.get("query") or args.get("q") or prompt[:60]
                    except Exception:
                        query = prompt[:60]
                    # We have detected a tool call — stop streaming current tokens that are tool syntax
                    # Emit tool_call instead of the raw tokens that contain the tool markup
                    # First, strip the tool call markup from already yielded text (fronted will hide it)
                    yield {"type": "tool_call", "name": tool_name, "arguments": {"query": query}, "query": query, "raw": m.group(0)}
                    t_tool = time.time()
                    try:
                        search_res = await web_search(query, count=5)
                        formatted = format_results(search_res["results"])
                        yield {"type": "tool_result", "name": tool_name, "result": search_res, "formatted": formatted, "latency_ms": int((time.time()-t_tool)*1000)}
                        # Generate second turn with results
                        augmented = f"Web search results for '{query}':\n{formatted}\n\nUser original: {prompt}\nAnswer concisely, voice style."
                        if self.mock:
                            top = search_res["results"][0] if search_res["results"] else {"content": "no results"}
                            mock_answer = f"Based on SearXNG ({search_res['source']}) for '{query}': {top['title']} — {top['content'][:140]}"
                            text_so_far = ""
                            for tok in mock_answer.split(" "):
                                await asyncio.sleep(0.018)
                                text_so_far += (" " if text_so_far else "") + tok
                                yield {"type": "llm_token", "token": " "+tok if text_so_far else tok, "text_so_far": text_so_far, "latency_ms": int((time.time()-t0)*1000)}
                            yield {"type": "llm_done", "text": text_so_far}
                            return
                        async for tok2 in self.generate_stream(augmented, max_new_tokens=max_new_tokens):
                            yield tok2
                        return
                    except Exception as e:
                        logger.exception(f"tool exec failed {e}")
                        yield {"type": "tool_result", "name": tool_name, "error": str(e)}
                        yield ev  # yield the original token that triggered it? but we already consumed
                        continue
                # Also check JSON tool pattern without <tool> wrapper
                jm = JSON_TOOL_RE.search(buffered)
                if jm and not saw_tool_call and '"query"' in buffered and web_search is not None and "tool_call" not in buffered:
                    # Try parse json
                    try:
                        import re
                        import json as js
                        # extract first json object
                        start = buffered.find('{')
                        depth=0
                        end=-1
                        for idx,ch in enumerate(buffered[start:]):
                            if ch=='{':
                                depth+=1
                            elif ch=='}':
                                depth-=1
                            if depth==0:
                                end=start+idx+1
                                break
                        if end!=-1:
                            obj=js.loads(buffered[start:end])
                            if obj.get("tool")=="web_search" and obj.get("query"):
                                saw_tool_call=True
                                query=obj["query"]
                                yield {"type": "tool_call", "name": "web_search", "arguments": {"query": query}}
                                t_tool=time.time()
                                search_res=await web_search(query, count=5)
                                formatted=format_results(search_res["results"])
                                yield {"type": "tool_result", "name": "web_search", "result": search_res, "formatted": formatted, "latency_ms": int((time.time()-t_tool)*1000)}
                                augmented=f"Search results for '{query}':\n{formatted}\n\nUser: {prompt}\nAnswer:"
                                async for tok2 in self.generate_stream(augmented):
                                    yield tok2
                                return
                    except Exception:
                        pass
                # Normal token streaming
                yield ev
            elif ev["type"] == "llm_done":
                if not saw_tool_call:
                    yield ev
                return
        # Fallback yield done
        yield {"type": "llm_done", "text": buffered}
