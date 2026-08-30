"""
Shared plumbing for the three agent harnesses (qwen_harness.py, harness.py,
pydantic_harness.py). Each wraps a single, module-level agent instance
(Assistant / ToolCallingAgent / PydanticAI Agent) that is reused across every
session and turn — factored out here since all three needed the identical fix
for the identical reason: a shared instance can be entered concurrently by
independent calls (two sessions chatting at once, or a barge-in'd turn's
orphaned background thread — asyncio.to_thread can't be interrupted mid-call —
still finishing when a new turn starts).
"""
import contextvars
import threading

# Per-call (loop, queue) target for tool_call/tool_result events, NOT a mutable
# global: a global "last attach() wins" singleton would silently deliver one
# call's events into a different concurrent call's WS queue. asyncio.to_thread()
# copies the current contextvars context into its worker thread, so setting this
# per-call (before to_thread) and reading it from the tool .call()/function
# methods (which execute on that thread) keeps each call's events routed
# correctly even when two calls are in flight together.
_emit_ctx: "contextvars.ContextVar[tuple | None]" = contextvars.ContextVar("_emit_ctx", default=None)


def set_emit_target(loop, event_q) -> None:
    _emit_ctx.set((loop, event_q))


def emit(ev: dict) -> None:
    ctx = _emit_ctx.get()
    if ctx:
        loop, q = ctx
        if loop and q:
            loop.call_soon_threadsafe(q.put_nowait, ev)


# Serializes the actual agent.run()/run_sync() invocation across concurrent
# calls into one harness's shared instance. These libraries' internal state
# (conversation memory, and in smolagents' case an explicit `reset=True` that
# wipes it at the start of every call) is not documented or verified safe under
# concurrent access from multiple threads on one instance — two overlapping
# calls could otherwise corrupt or cross-contaminate each other's state (e.g.
# one call's reset wiping out another's in-progress run). This trades a small
# amount of latency (a second call waits for an overlapping first one to finish
# — rare: only when an orphaned barged-in call is still running) for guaranteed
# correctness regardless of what the underlying library actually does.
agent_call_lock = threading.Lock()
