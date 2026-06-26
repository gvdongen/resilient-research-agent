"""Demo 2 — the controller in front of the research agent (a Virtual Object).

It tracks the in-flight run by the invocation id it gets back when it sends the run
(stored in its own state — known synchronously, so concurrent messages never race).
A message arrives. If nothing is running, start a run. Otherwise a small LLM classifier
picks how to handle it, and each choice is one Restate primitive:

- enqueue   — just send another run; the research Virtual Object serializes, so it
              runs after the current one. No queue to manage.
- interrupt — `ctx.cancel_invocation(...)` the running run, then start a fresh one.
- steer     — `ctx.resolve_signal(...)` pushes the message into the running loop.
"""

import restate

from llm import FAST_MODEL, chat_structured
from research import run
from schemas import StrategyChoice

CLASSIFIER = """A research run is already in progress and a new message arrived. Pick:
- "steer": refines/adds to the SAME goal (a constraint, a narrower focus) — fold it in.
- "interrupt": changes the goal (different topic, or "stop/forget that") — restart.
- "enqueue": a separate follow-up to run AFTER the current one finishes."""


async def classify(goal: str, message: str) -> str:
    out = await chat_structured(
        [
            {"role": "system", "content": CLASSIFIER},
            {"role": "user", "content": f"Current goal:\n{goal}\n\nNew message:\n{message}"},
        ],
        StrategyChoice,
        model=FAST_MODEL,
    )
    return out["strategy"]


controller = restate.VirtualObject("Controller")


async def _start(ctx: restate.ObjectContext, text: str) -> None:
    """Start a run and remember its invocation id. (Enqueue is just this while one is
    already running: the research VO serializes, so the new run waits its turn.)"""
    handle = ctx.object_send(run, key=ctx.key(), arg=text)
    ctx.set("current", {"id": await handle.invocation_id(), "goal": text})


@controller.handler()
async def message(ctx: restate.ObjectContext, text: str) -> str:
    # Retrieve history
    msgs = await restate.get("history", type_hint=list) or [system(SYSTEM)]
    msgs.append(user(query))

    current = await ctx.get("current", type_hint=dict)
    if current is None:
        await _start(ctx, text)
        return "started research"

    match await ctx.run_typed("classify", classify, goal=current["goal"], message=text):
        case "interrupt":
            ctx.cancel_invocation(current["id"])
            await _start(ctx, text)
            return "interrupted; restarted with the new goal"
        case "steer":
            ctx.resolve_signal(current["id"], "steer", text)
            return "steering the running research"


@controller.handler()
async def done(ctx: restate.ObjectContext, inv_id: str) -> None:
    """Called by the research agent when a run finishes. Clear our pointer — but only
    if it's still the run we're tracking (a newer run may have replaced it)."""
    current = await ctx.get("current", type_hint=dict)
    if current and current["id"] == inv_id:
        ctx.clear("current")
