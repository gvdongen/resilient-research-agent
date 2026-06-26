"""Demo 2 — the durable research agent (a Virtual Object, keyed by session).

The simplest agent loop: call the LLM, run the web-search tools it asks for, repeat
until it writes its report. Every LLM call and every tool call is a durable step
(`ctx.run_typed`), so a crash replays from the journal instead of re-running (re-paying)
them. Because it's a Virtual Object, runs for one session execute one-at-a-time — so a
second message just queues behind the running one (that's "enqueue", for free).
"""

from datetime import timedelta

import restate as rst
from restate import ObjectContext, RestateDurableFuture, Context

from llm import call_llm, system, user, tool_result
from tools import websearch_tool, websearch

MAX_TURNS = 6

SYSTEM = """You are a research analyst with a web_search tool. Investigate the user's
request: search a few angles, then write a concise report of the key findings with
source URLs."""

agent = rst.Service("ResearchAgent")


@agent.handler()
async def research_agent(restate: Context, msgs: str) -> str:

    # Agent loop
    while True:
        msgs += await steers(restate)

        # LLM call - persisted and recovered
        result = await restate.run_typed("llm", call_llm, messages=msgs, tools=[websearch_tool])
        msgs.append(result.message)

        # End of loop
        if not result.tool_calls:
            break

        # Tool calls - persisted and recovered
        searches = [
            restate.run_typed("web_search", websearch, arguments=tc.arguments)
            for tc in result.tool_calls
        ]
        await rst.gather(*searches)
        msgs += await format_results(result, searches)


    from controller import done
    restate.object_send(done, key=restate.key(), arg=restate.request().id)

    return result.response








async def format_results(result, searches: list[RestateDurableFuture[dict]]) -> list[dict]:
    return [tool_result(tc, str(await s)) for tc, s in zip(result.tool_calls, searches)]


async def steers(ctx: restate.ObjectContext) -> list[dict]:
    steer = ctx.signal("steer", type_hint=str)
    timeout = ctx.sleep(timedelta(0))
    match await restate.select(steer=steer, now=timeout):
        case ["steer", update]:
            return [user(f"Steering update: {update}")]
        case _:
            return []
