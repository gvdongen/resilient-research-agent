"""Phase 1 — Durable single-shot research.

`SimpleResearchAgent.search(query)` runs a LangChain agent loop with
three Tavily tools. `RestateMiddleware` journals every LLM response;
each `@tool` wraps its Tavily call in `restate_context().run_typed(...)`,
so retries replay from the journal instead of re-hitting the API.

Stateless. One query in, one structured `Research` answer out.
"""

import restate
from langchain.agents import create_agent
from langchain_core.tools import tool
from restate.ext.langchain import RestateMiddleware, restate_context
from utils.schemas import Research
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


researcher = create_agent(
    model="openai:gpt-5",
    tools=[web_search, extract_urls, crawl_sites],
    system_prompt="""You are a research assistant. Use web_search, extract_urls,
    and crawl_sites to investigate the user's question. Keep the loop tight —
    at most 3 rounds of tool calls. Return a concise summary and a list of
    source URLs. Cite only sources you actually read.""",
    response_format=Research,
    middleware=[RestateMiddleware()],
)


agent = restate.Service("SimpleResearchAgent")


@agent.handler()
async def search(_ctx: restate.Context, query: str) -> Research:
    result = await researcher.ainvoke({"messages": query})
    return result["structured_response"]
