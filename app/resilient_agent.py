"""Phase 1 (LangChain) — Durable research agent, single query.

Uses LangChain's `create_agent` with Restate's `RestateMiddleware` for
journaled LLM responses. Tools wrap their work in `restate_context().run_typed`
so tool results are also journaled.
"""

import restate
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, ConfigDict
from restate.ext.langchain import RestateMiddleware, restate_context
from tavily import TavilyClient
from typing import Literal
Range = Literal["day", "week", "month", "year"]


# ---- Schemas ---------------------------------------------------------

class ChatHistory(BaseModel):
    """Type hint for the messages list persisted in object state."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: list = []


class Research(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    sources: list[str]


# ---- Tavily-backed tools ----------------------------------------------------

tavily_client = TavilyClient()


@tool
async def web_search(query: str, time_range: Range = "month") -> dict:
    """Search the web via Tavily. time_range is one of: day, week, month, year."""

    async def _call() -> dict:
        return tavily_client.search(query=query, time_range=time_range, search_depth="advanced", topic="general")

    return await restate_context().run_typed(f"web_search: {query}", _call)


@tool
async def extract_urls(urls: list[str]) -> dict:
    """Return the full readable text of a list of web pages."""

    async def _call() -> dict:
        return tavily_client.extract(urls=urls, extract_depth="advanced")

    return await restate_context().run_typed(f"extract: {urls}", _call)


@tool
async def crawl_site(url: str, instructions: str = "") -> dict:
    """Crawl a website (guided by natural-language instructions) and return content from up to 10 pages."""

    async def _call() -> dict:
        return tavily_client.crawl(url=url, instructions=instructions)

    return await restate_context().run_typed(f"crawl: {url}", _call)


# ---- Agent ----------------------------------------------------

_agent = create_agent(
    model=init_chat_model("openai:gpt-5"),
    tools=[web_search, extract_urls, crawl_site],
    system_prompt="""You are a research assistant with access to the web via three tools: 
    `web_search` for finding sources (use `time_range` for recency), 
    `extract_url` for reading a single page in full, and `crawl_site` for 
    covering a documentation or news site. Always cite the URLs you used. 
    Be concise and factual.""",
    response_format=Research,
    middleware=[RestateMiddleware()],
)


agent = restate.Service("ResilientResearchAgent")


@agent.handler()
async def search(_ctx: restate.Context, query: str) -> Research:
    result = await _agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["structured_response"]
