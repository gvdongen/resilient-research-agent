"""Phase 2 — Durable chat session with a research agent.

`ResearchSession` is a Virtual Object keyed by session_id. The exclusive
`chat(query)` handler appends the user message, runs a LangChain agent
with the *full* prior history as context, and persists the updated
message list back to state.

Concurrent `chat` calls on the same session serialize automatically — no
race conditions on the conversation log. State survives crashes.
"""

import restate
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from restate.ext.langchain import restate_context, RestateMiddleware
from utils.schemas import ChatHistory
from utils.tools import tavily_search, tavily_extract, tavily_crawl, Range

@tool
async def web_search(queries: list[str], time_range: Range = "month") -> list[dict]:
    """Search the web with a list of queries. time_range is one of: day, week, month, year."""
    searches = [
        restate_context().run_typed(f"web_search:{q}", tavily_search, query=q, range=time_range)
        for q in queries
    ]
    await restate.gather(*searches)
    return [await result for result in searches]


@tool
async def extract_urls(urls: list[str]) -> dict:
    """Return the full readable text of a list of web pages."""
    return await restate_context().run_typed("extract_urls", tavily_extract, urls=urls)


@tool
async def crawl_sites(urls: list[str], instructions: str = "") -> list[dict]:
    """Crawl a website (guided by natural-language instructions) and return content from up to 10 pages."""
    crawls = [
        restate_context().run_typed(f"crawl_site:{u}", tavily_crawl, url=u, instructions=instructions)
        for u in urls
    ]
    await restate.gather(*crawls)
    return [await result for result in crawls]


# ----------- Agent + VO ---------------------

chat_researcher = create_agent(
    model="openai:gpt-5",
    tools=[web_search, extract_urls, crawl_sites],
    system_prompt="""You are a research assistant. Use web_search, extract_urls,
    and crawl_site to answer the user's questions. Keep the loop tight —
    at most 3 rounds of tool calls per user message. Cite sources inline as
    URLs. Carry context across turns — earlier messages in the conversation
    may already contain relevant background.""",
    middleware=[RestateMiddleware()],
)


agent = restate.VirtualObject("ResearchSession")


@agent.handler()
async def chat(restate: restate.ObjectContext, query: str) -> str:
    history = await restate.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append(HumanMessage(content=query))

    result = await chat_researcher.ainvoke({"messages": history.messages})

    restate.set("messages", ChatHistory(messages=result["messages"]))
    return result["messages"][-1].content


@agent.handler(kind="shared")
async def get_history(ctx: restate.ObjectSharedContext) -> ChatHistory:
    """Read-only — inspect what this session has said so far."""
    return await ctx.get("messages", type_hint=ChatHistory) or ChatHistory()
