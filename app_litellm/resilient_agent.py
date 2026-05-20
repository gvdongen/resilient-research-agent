"""Phase 1 — Durable research agent (single query).

Stateless: one query in, one structured Research answer out. Every LLM call
and every tool call is wrapped in `ctx.run_typed`, so Restate journals each
step and replays the loop deterministically on resume. When the LLM emits
multiple tool calls in one turn, they run in parallel via `restate.gather`.
"""

import json
import restate
from tavily import TavilyClient
from litellm import acompletion
from litellm.types.utils import ModelResponse
from litellm.utils import function_to_dict
from pydantic import BaseModel, ConfigDict
from typing import Literal
Range = Literal["day", "week", "month", "year"]

# ---- Tavily-backed tools ----------------------------------------------------

tavily_client = TavilyClient()

async def web_search(query: str, time_range: Range = "month") -> dict:
    """Search the web via Tavily. time_range is one of: day, week, month, year."""
    return tavily_client.search(query=query, time_range=time_range, search_depth="advanced", topic="general")


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


# ---- Output schemas ---------------------------------------------------------

class Research(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    sources: list[str]


# ---- System prompt + tool specs ---------------------------------------------

SYSTEM_RESEARCH = (
    "You are a research assistant with access to the web via three tools: "
    "`web_search` for finding sources (use `time_range` for recency), "
    "`extract_url` for reading a single page in full, and `crawl_site` for "
    "covering a documentation or news site. Always cite the URLs you used. "
    f"Be concise and factual."
)


# ---- Agent ----------------------------------------------------

agent = restate.Service("ResilientResearchAgent")


@agent.handler()
async def search(ctx: restate.Context, query: str) -> Research:
    messages = [
        {"role": "system", "content": SYSTEM_RESEARCH},
        {"role": "user", "content": query},
    ]

    while True:
        # Call the LLM with a durable step
        async def call_llm() -> ModelResponse:
            return await acompletion(
                model="gpt-5-mini", messages=messages, tools=TOOL_SPECS, response_format=Research
            )
        response = await ctx.run_typed("llm", call_llm)
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        # Return result, if no tool calls
        if not msg.tool_calls:
            return Research.model_validate_json(msg.content)

        # Execute tool calls in durable step
        handles = [
            ctx.run_typed(tc.function.name, TOOLS[tc.function.name], **json.loads(tc.function.arguments))
            for tc in msg.tool_calls
        ]
        await restate.gather(*handles)
        for tc, h in zip(msg.tool_calls, handles):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(await h),
            })

