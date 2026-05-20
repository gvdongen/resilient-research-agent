"""DeepResearchAgent — autonomous daily loop, LangChain edition.

Each day:
  1. NewsScoutAgent scans the news on a topic and writes a digest.
  2. Post the digest to Slack with a curl to resolve an awakeable to request
     a deep dive on a news topic. The orchestrator suspends on `restate.select`
     until the awakeable fires or a 24h timeout elapses.
  3. If a deep-dive topic arrives →
     PlannerAgent → N parallel ResearcherAgents → WriterAgent
     and then the FinalReport is posted to Slack.
  4. Self-schedule tomorrow's run via `ctx.service_send(..., send_delay=days=1)`.
"""

from datetime import timedelta
from typing import Literal

import restate
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from restate.ext.langchain import RestateMiddleware
from utils.schemas import (
    DailyResult,
    FinalReport,
    NewsDigest,
    ResearchPlan,
    SubReport,
)
from utils.tools import summarize, to_brief, post_news, post_report
from langchain_core.tools import tool
from restate.ext.langchain import restate_context
from tavily import TavilyClient

Range = Literal["day", "week", "month", "year"]


# ----------- Tools ---------------------

tavily_client = TavilyClient()


@tool
async def web_search(query: str, time_range: Range = "month") -> dict:
    """Search the web via Tavily. time_range is one of: day, week, month, year."""
    def _search() -> dict:
        return tavily_client.search(
            query=query, time_range=time_range, search_depth="advanced", topic="general"
        )
    return await restate_context().run_typed(f"web_search:{query}", _search)


@tool
async def extract_urls(urls: list[str]) -> dict:
    """Return the full readable text of a list of web pages."""
    def _extract() -> dict:
        return tavily_client.extract(urls=urls, extract_depth="advanced")
    return await restate_context().run_typed("extract_urls", _extract)


@tool
async def crawl_site(url: str, instructions: str = "") -> dict:
    """Crawl a website (guided by natural-language instructions) and return content from up to 10 pages."""
    def _crawl() -> dict:
        return tavily_client.crawl(url=url, instructions=instructions)
    return await restate_context().run_typed(f"crawl_site:{url}", _crawl)


# ----------- Agents ---------------------

news_scout = create_agent(
    model=init_chat_model("openai:gpt-5"),
    tools=[web_search, extract_urls],
    system_prompt="""You are a news scout. Given a topic, use `web_search` with
    `time_range='day'` to find what's new in the last 24 hours. If a
    story looks important or unclear, follow up with `extract_urls` to
    read it in full. Return a NewsDigest: a one-paragraph overview plus
    3-8 distinct news items (headline, 1-2 sentence summary, source URL).
    Skip anything you already covered earlier.""",
    response_format=NewsDigest,
    middleware=[RestateMiddleware()],
)

planner = create_agent(
    model=init_chat_model("openai:gpt-5"),
    system_prompt="""You are a senior research planner. Given a topic and a digest of
    today's news on it, produce a tight research plan: a short rationale
    (what's worth digging into and why) plus 3-5 sharply scoped subtopics.
    Each subtopic should be a self-contained research question that a
    separate researcher can investigate in parallel without overlap with
    the others. Prefer subtopics that dig into the most consequential
    items from today's news.""",
    response_format=ResearchPlan,
    middleware=[RestateMiddleware()],
)

researcher = create_agent(
    model=init_chat_model("openai:gpt-5"),
    tools=[web_search, extract_urls, crawl_site],
    system_prompt="""You are a focused research analyst. You have web_search, extract_urls,
    and crawl_site available. Investigate the assigned subtopic thoroughly:
    search with recency-appropriate time_range, then read the most
    promising sources in full. Cite every claim with a URL. Stop as soon as
    you have enough to write a tight 200-400 word findings section.""",
    response_format=SubReport,
    middleware=[RestateMiddleware()],
)

writer = create_agent(
    model=init_chat_model("openai:gpt-5"),
    system_prompt="""You are a senior editor turning raw research notes into a polished
    report. Take the topic, the plan's rationale, and the per-subtopic
    findings. Produce: a sharp headline, a 3-4 sentence executive summary,
    one section per subtopic (heading + 150-250 word body in markdown),
    and a de-duplicated list of source URLs. Cite inline where it adds
    credibility. Do not invent facts beyond what the findings contain.""",
    response_format=FinalReport,
    middleware=[RestateMiddleware()],
)

# ----------- Agent services ---------------------

news_scout_agent = restate.Service("NewsScoutAgent")


@news_scout_agent.handler()
async def scan(_ctx: restate.Context, topic: str) -> NewsDigest:
    result = await news_scout.ainvoke({"messages": f"Topic: {topic}"})
    return result["structured_response"]


planner_agent = restate.Service("PlannerAgent")


@planner_agent.handler()
async def plan(_ctx: restate.Context, brief: str) -> ResearchPlan:
    result = await planner.ainvoke({"messages": brief})
    return result["structured_response"]


research_agent = restate.Service("ResearchAgent")


@research_agent.handler()
async def investigate(_ctx: restate.Context, subtopic: str) -> SubReport:
    result = await researcher.ainvoke({"messages": f"Subtopic: {subtopic}"})
    return result["structured_response"]


writer_agent = restate.Service("WriterAgent")


@writer_agent.handler()
async def write(_ctx: restate.Context, brief: str) -> FinalReport:
    result = await writer.ainvoke({"messages": brief})
    return result["structured_response"]


# ----------- Orchestration ---------------------

agent = restate.Service("NewsResearchAgent")


@agent.handler()
async def run(ctx: restate.Context, topic: str) -> DailyResult:
    # Self-schedule tomorrow
    ctx.service_send(run, arg=topic, send_delay=timedelta(days=1))

    # Stage 1 — daily news scan
    news = await ctx.service_call(scan, arg=topic)

    # Stage 2 — send news digest to Slack, ask for a deep-dive topic
    awk_id, decision_promise = ctx.awakeable(type_hint=str)
    await ctx.run_typed("slack-news-update", post_news, topic=topic, digest=news, awk_id=awk_id)

    match await restate.select(
        decision=decision_promise,
        timeout=ctx.sleep(timedelta(days=1)),
    ):
        case ["timeout"]:
            return DailyResult(news=news)
        case ["decision", answer]:
            deep_dive_topic = answer

    # Stage 3 — plan
    research_plan = await ctx.service_call(plan, arg=summarize(deep_dive_topic, news))

    # Stage 4 — fan out one Researcher per subtopic, in parallel
    handles = [
        ctx.service_call(investigate, arg=sub)
        for sub in research_plan.subtopics
    ]
    await restate.gather(*handles)
    sub_reports = [await h for h in handles]

    # Stage 5 — synthesize the final report
    report = await ctx.service_call(write, arg=to_brief(deep_dive_topic, research_plan, sub_reports))

    # Stage 6 — deliver the report
    await ctx.run_typed("slack-topic-report", post_report, topic=deep_dive_topic, report=report)
    return DailyResult(news=news, report=report)
