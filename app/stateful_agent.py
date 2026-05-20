"""Phase 2 (LangChain) — Stateful research agent, conversation persists.

VirtualObject keyed by user_id. Message history is stored in object state, so
follow-up queries on the same key carry the conversation. Concurrent calls on
the same key serialize automatically.
"""

import restate
from langchain_core.messages import HumanMessage
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


agent = restate.VirtualObject("StatefulResearchAgent")


@agent.handler()
async def ask(ctx: restate.ObjectContext, query: str) -> Research:
    history = await ctx.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append(HumanMessage(content=query))

    result = await _agent.ainvoke({"messages": history.messages})

    ctx.set("messages", ChatHistory(messages=result["messages"]))
    return result["structured_response"]


@agent.handler(kind="shared")
async def get_history(ctx: restate.ObjectSharedContext) -> ChatHistory:
    return await ctx.get("messages", type_hint=ChatHistory) or ChatHistory()
