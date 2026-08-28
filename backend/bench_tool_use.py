#!/usr/bin/env python3
"""Tool-use benchmark: Ling 3.0 tiny vs Qwen3.5-2B vs Apodex-1.0-2B-SFT.
Native OpenAI tools API via llama-server (same runtime, fair). Two-pass:
  1) user prompt + tools=[web_search] -> expect tool_call JSON
  2) append tool result -> expect coherent final answer
Measures: tool_call validity, arg query, TTFT, tok/s, answer quality.
"""
import asyncio, json, time, httpx, sys

PROMPTS = [
    {"id": "weather", "q": "What is the weather in Paris today?"},
    {"id": "france",  "q": "Who is the president of France?"},
    {"id": "ai_news", "q": "What are the latest AI news in 2026?"},
    {"id": "py314",   "q": "What are the new features in Python 3.14?"},
    {"id": "tokyo",   "q": "東京の今日の天気は？"},
]

TOOLS = [{"type": "function", "function": {
    "name": "web_search",
    "description": "Search the web for current info. Use for weather, news, facts, recent events.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

CANON = {"Paris": 18, "France": "Emmanuel Macron", "AI": "2026 AI news", "Python": "free-threading", "Tokyo": 22}
SNIPPET = "SearXNG result: Paris 18C sunny. Emmanuel Macron is president of France (since 2017). Python 3.14 free-threading, 2026 AI: agents + streaming TTS. Tokyo 22C."

MODELS = {
    "ling-3-tiny":    ("127.0.0.1:11435", "ling-3-tiny"),
    "qwen3.5-2b":     ("127.0.0.1:11436", "qwen3.5-2b"),
    "apodex-2b-sft":  ("127.0.0.1:11437", "apodex-2b"),
}

async def chat(base, model, messages, max_tokens=256):
    t0=time.time(); first=None; text=""; tcalls=[]; ntok=0
    payload={"model":model,"messages":messages,"tools":TOOLS,"tool_choice":"auto",
             "stream":True,"max_tokens":max_tokens,"temperature":0.2}
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"http://{base}/v1/chat/completions", json=payload) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "): continue
                d=line[6:].strip()
                if d=="[DONE]": break
                try:
                    j=json.loads(d)
                    if first is None: first=time.time()-t0
                    ch=j["choices"][0]
                    delta=ch.get("delta",{})
                    ntok+=1
                    if delta.get("content"): text+=delta["content"]
                    for tc in (delta.get("tool_calls") or []):
                        tcalls.append(tc)
                except Exception: pass
    ttft=int((first or 0)*1000)
    # merge tool calls (streaming splits) + extract final JSON
    merged={}
    for tc in tcalls:
        i=tc.get("index",0)
        fn=tc.get("function",{})
        m=merged.setdefault(i,{"name":"","arguments":""})
        m["name"]+=fn.get("name","")
        m["arguments"]+=fn.get("arguments","")
    tool_calls=[]; args=None
    for i,m in merged.items():
        tool_calls.append({"name":m["name"],"arguments":m["arguments"]})
        if "web_search" in m["name"]:
            try: args=json.loads(m["arguments"])
            except Exception: args={"query":m["arguments"][:60]}
    tok_s = ntok/max((time.time()-t0),0.001)
    return {"text":text.strip(), "tool_calls":tool_calls, "args":args, "ttft":ttft, "tok_s":round(tok_s,1), "total_s":round(time.time()-t0,2)}

async def bench_one(model_key, base, alias):
    rows=[]
    for p in PROMPTS:
        try:
            r1 = await chat(base, alias, [{"role":"user","content":p["q"]}])
            tc = r1["tool_calls"]
            valid = bool(tc and tc[0]["name"]=="web_search")
            # pass 2: feed tool result, ask to answer
            msgs=[{"role":"user","content":p["q"]}]
            if valid:
                aidx=tc[0].get("index",0)
                msgs.append({"role":"assistant","content":None,
                             "tool_calls":[{"id":"call_0","type":"function","function":{"name":"web_search","arguments":json.dumps(r1["args"] or {"query":p["q"]})}}]})
                msgs.append({"role":"tool","tool_call_id":"call_0","content":SNIPPET+" Source: SearXNG self-hosted."})
            r2 = await chat(base, alias, msgs)
            ans = r2["text"]
            answer_ok = len(ans)>=30 and any(k in ans.lower() for k in ["paris","macron","python","ai","tokyo","18","22"])
            rows.append({"id":p["id"],"tool_call_ok":valid,"query":(r1["args"] or {}).get("query","")[:40],
                         "ttft1":r1["ttft"],"tok_s1":r1["tok_s"],"ttft2":r2["ttft"],"tot2":r2["total_s"],
                         "answer_ok":answer_ok,"ans":ans[:70]})
        except Exception as e:
            rows.append({"id":p["id"],"err":str(e)[:60]})
    # aggregate
    n=len([r for r in rows if "err" not in r])
    tc_ok=sum(1 for r in rows if r.get("tool_call_ok"))
    ans_ok=sum(1 for r in rows if r.get("answer_ok"))
    ttfts=[r["ttft1"] for r in rows if "ttft1" in r]
    print(f"\n### {model_key}")
    print(f"tool_call_ok {tc_ok}/{n} | answer_ok {ans_ok}/{n} | TTFT1 avg {sum(ttfts)//len(ttfts) if ttfts else 0}ms")
    for r in rows:
        if "err" in r: print("  ", r["id"], "ERR", r["err"]); continue
        print(f"  {r['id']:8s} tc={'Y' if r['tool_call_ok'] else 'N'} ans={'Y' if r['answer_ok'] else 'N'} "
              f"ttft1 {r['ttft1']:>4}ms tok/s {r['tok_s1']:>5} q=\"{r['query']}\"")
        print(f"            ans: {r['ans']}")

async def main():
    for key,(base,alias) in MODELS.items():
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r=await c.get(f"http://{base}/v1/models")
            if r.status_code!=200: print(f"{key} not up"); continue
        except Exception:
            print(f"{key} not up on {base}"); continue
        await bench_one(key, base, alias)

if __name__=="__main__":
    asyncio.run(main())