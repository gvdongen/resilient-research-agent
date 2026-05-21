"""Phase 2 — Durable chat session with a research agent (custom litellm loop).

`ResearchSession` is a Virtual Object keyed by session_id. The exclusive
`chat(query)` handler appends the user message, runs the tool-calling loop
with the *full* prior history as context, and persists the updated message
list back to state.

Concurrent `chat` calls on the same session serialize automatically — no
race conditions on the conversation log. State survives crashes.
"""

import json
from typing import Literal

import restate
from litellm import acompletion
from litellm.types.utils import ModelResponse
from litellm.utils import function_to_dict
from tavily import TavilyClient

Range = Literal["day", "week", "month", "year"]


# ----------- Tools ---------------------

tavily_client = TavilyClient()


async def web_search(query: str, time_range: Range = "month") -> dict:
    """Search the web via Tavily. time_range is one of: day, week, month, year."""
    return tavily_client.search(
        query=query, time_range=time_range, search_depth="advanced", topic="general"
    )


async def extract_urls(urls: list[str]) -> dict:
    """Return the full readable text of a list of web pages."""
    return tavily_client.extract(urls=urls, extract_depth="advanced")


async def crawl_site(url: str, instructions: str = "") -> dict:
    """Crawl a website (guided by natural-language instructions) and return content from up to 10 pages."""
    return tavily_client.crawl(url=url, instructions=instructions)


TOOL_SPECS = [
    {"type": "function", "function": function_to_dict(fn)}
    for fn in (web_search, extract_urls, crawl_site)
]

TOOLS = {
    "web_search": web_search,
    "extract_urls": extract_urls,
    "crawl_site": crawl_site,
}


SYSTEM = (
    "You are a research assistant. Use web_search, extract_urls, and "
    "crawl_site to answer the user's questions thoroughly. Cite sources "
    "inline as URLs. Carry context across turns — earlier messages in the "
    "conversation may already contain relevant background."
)

MAX_TURNS = 3


# ----------- Virtual Object ---------------------

agent = restate.VirtualObject("ResearchSession")


@agent.handler()
async def chat(ctx: restate.ObjectContext, query: str) -> str:
    # Load history from state (durable). On first turn, seed with the system prompt.
    messages = await ctx.get("messages") or [{"role": "system", "content": SYSTEM}]
    messages.append({"role": "user", "content": query})

    for _ in range(MAX_TURNS):
        # Durable LLM call — no response_format, this is free-form chat
        async def call_llm() -> ModelResponse:
            return await acompletion(
                model="gpt-5",
                messages=messages,
                tools=TOOL_SPECS,
            )

        response = await ctx.run_typed("llm", call_llm)
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # No more tool calls → save history, return assistant's text
        if not msg.tool_calls:
            ctx.set("messages", messages)
            return msg.content or ""

        # Durable parallel tool calls
        handles = [
            ctx.run_typed(
                tc.function.name,
                TOOLS[tc.function.name],
                **json.loads(tc.function.arguments),
            )
            for tc in msg.tool_calls
        ]
        await restate.gather(*handles)
        for tc, h in zip(msg.tool_calls, handles):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(await h),
                }
            )

    # Budget exhausted — force a final answer with no more tools
    messages.append(
        {
            "role": "user",
            "content": "Turn budget exhausted. Answer the user now with what you have.",
        }
    )

    async def call_llm_final() -> ModelResponse:
        return await acompletion(model="gpt-5", messages=messages)

    response = await ctx.run_typed("llm-final", call_llm_final)
    final_msg = response.choices[0].message
    messages.append(final_msg.model_dump(exclude_none=True))
    ctx.set("messages", messages)
    return final_msg.content or ""


@agent.handler(kind="shared")
async def get_history(ctx: restate.ObjectSharedContext) -> list[dict]:
    """Read-only — inspect what this session has said so far."""
    return await ctx.get("messages") or []
