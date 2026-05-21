"""Phase 1 — Durable single-shot research (custom litellm loop).

`SimpleResearchAgent.search(query)` runs a hand-written tool-calling
loop against `litellm.acompletion`. Each LLM call and each Tavily tool
call is wrapped in `ctx.run_typed(...)`, so retries replay from the
journal instead of re-executing.

Stateless. One query in, one structured `Research` answer out.
"""

import json
from typing import Literal

import restate
from litellm import acompletion
from litellm.types.utils import ModelResponse
from litellm.utils import function_to_dict
from tavily import TavilyClient

from utils.schemas import Research

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
    "crawl_site to investigate the user's question. Return a concise summary "
    "and a list of source URLs. Cite only sources you actually read."
)

MAX_TURNS = 3


# ----------- Service ---------------------

agent = restate.Service("SimpleResearchAgent")


@agent.handler()
async def search(ctx: restate.Context, query: str) -> Research:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]

    for _ in range(MAX_TURNS):
        # Durable LLM call
        async def call_llm() -> ModelResponse:
            return await acompletion(
                model="gpt-5",
                messages=messages,
                tools=TOOL_SPECS,
                response_format=Research,
            )

        response = await ctx.run_typed("llm", call_llm)
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # No more tool calls → final answer
        if not msg.tool_calls:
            return Research.model_validate_json(msg.content)

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

    # Budget exhausted — force a final structured answer with no more tools
    messages.append(
        {
            "role": "user",
            "content": "Turn budget exhausted. Return the Research result now with what you have.",
        }
    )

    async def call_llm_final() -> ModelResponse:
        return await acompletion(
            model="gpt-5",
            messages=messages,
            response_format=Research,
        )

    response = await ctx.run_typed("llm-final", call_llm_final)
    return Research.model_validate_json(response.choices[0].message.content)
