"""The reusable durable-agent engine.

Two pieces the application (deep_research.py) builds on:
  • LLMGateway — every model call goes through one governed service: a model
    allow-list (policy) + a scope so `restate rules set llm --concurrency N`
    caps concurrent model calls org-wide.
  • run_agent — a hand-written tool loop where every model and tool call is a
    durable Restate step, so a crash replays from the journal.
"""

import json

import restate
from litellm.utils import function_to_dict

from .schemas import LLMRequest
from .tools import (
    Range,
    call_crawl_api,
    call_extract_api,
    call_websearch_api,
    provider_call,
    to_schema,
)

MODEL = "gpt-5"
FAST_MODEL = "gpt-5-mini"
APPROVED_MODELS = {"gpt-5", "gpt-5-mini"}


# ----------- Tools the researchers can call ---------------------


async def web_search(query: str, time_range: Range = "month") -> dict:
    """Search the web. time_range is one of: day, week, month, year."""
    return call_websearch_api(query=query, range=time_range)


async def extract_urls(urls: list[str]) -> dict:
    """Return the full readable text of a list of web pages."""
    return await call_extract_api(urls=urls)


async def crawl_site(url: str, instructions: str = "") -> dict:
    """Crawl a website (guided by natural-language instructions) and return content from up to 10 pages."""
    return call_crawl_api(url=url, instructions=instructions)


TOOL_SPECS = [
    {"type": "function", "function": function_to_dict(fn)}
    for fn in (web_search, extract_urls, crawl_site)
]
TOOLS = {"web_search": web_search, "extract_urls": extract_urls, "crawl_site": crawl_site}


# ----------- LLM Gateway — policy + flow control ---------------------

llm_gateway = restate.Service("LLMGateway")


@llm_gateway.handler()
async def complete(ctx: restate.Context, req: LLMRequest) -> dict:
    # 1. policy guardrail
    if req.model not in APPROVED_MODELS:
        raise restate.TerminalError(
            f"Model '{req.model}' is not on the approved list {sorted(APPROVED_MODELS)}"
        )

    # 2. call LLM
    response = await ctx.run_typed("provider", provider_call, req=req)
    return response["choices"][0]["message"]


async def durable_llm_call(
    ctx: restate.ObjectContext,
    *,
    model: str = MODEL,
    instructions: str | None = None,
    messages: list[dict],
    output,
    tools: list | None = None,
) -> dict:
    """One model call, sent through the "llm" scope (concurrency cap) to the gateway.
    Returns the raw assistant message dict so the loop reads it without reconstruction."""
    if instructions:
        messages=[{"role": "user", "content": instructions}, *messages]
    req = LLMRequest(model=model, messages=messages, tools=tools, output=to_schema(output))
    return await ctx.scope(ctx.key()).service_call(complete, arg=req)


# ----------- Durable agent loop ---------------------


async def run_agent(ctx: restate.ObjectContext, prompt: str, messages: list[dict], *, output, tools=None, max_turns=3):
    """Call the model (via the gateway), run any tools it asks for, repeat until it
    returns a structured answer. Every step is journaled, so a crash replays instead
    of re-running (re-paying for) the model and tool calls."""
    msgs = list(messages)
    for _ in range(max_turns):
        message = await durable_llm_call(ctx, instructions=prompt, messages=msgs, output=output, tools=tools)
        msgs.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return output.model_validate_json(message["content"])

        # Durable parallel tool calls
        handles = [
            ctx.run_typed(tc["function"]["name"], TOOLS[tc["function"]["name"]], **json.loads(tc["function"]["arguments"]))
            for tc in tool_calls
        ]
        await restate.gather(*handles)
        for tc, h in zip(tool_calls, handles):
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(await h)})

    # Budget exhausted — force a final structured answer with no more tools
    msgs.append({"role": "user", "content": "Turn budget exhausted. Return the result now with what you have."})
    message = await durable_llm_call(ctx, instructions=prompt, messages=msgs, output=output)
    return output.model_validate_json(message["content"])
