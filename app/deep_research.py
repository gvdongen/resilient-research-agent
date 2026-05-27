"""
Deep Research agent
- gets triggered by messages in a Slack channel
- kicks of planner, parallel research agents, and writer
- posts reply back in Slack channel

Option to daily scan news on a topic
"""

from datetime import timedelta
import restate
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from restate.ext.langchain import RestateMiddleware, restate_context
from utils.restate_chat_model import init_durable_model
from utils.schemas import *
from utils.tools import post_news, to_brief, post_plan, post_report
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
    system_prompt="""You are a focused research analyst. You have web_search, extract_urls,
    and crawl_site available. Investigate the assigned subtopic thoroughly:
    search with recency-appropriate time_range, then read the most
    promising sources in full. Keep the loop tight — at most 3 rounds of
    tool calls. Cite every claim swith a URL. Stop as soon as you have
    enough to write a tight 200-400 word findings section.""",
    response_format=Report,
    middleware=[RestateMiddleware()]
)

research_agent = restate.Service("ResearchAgent")


@research_agent.handler()
async def investigate(_rst: restate.Context, topic: str) -> Report:
    result = await researcher.ainvoke({"messages": f"Topic: {topic}"})
    return result["structured_response"]



# ----------- Deep Research Orchestrator ---------------------

planner = create_agent(
    model="openai:gpt-5",
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


writer = create_agent(
    model="openai:gpt-5",
    system_prompt="""You are a senior editor turning raw research notes into a polished
    report. Take the topic, the plan's rationale, and the per-subtopic
    findings. Write a concise report about 3-5 key findings (at most 100 words explanation).
    Add a de-duplicated list of the top 5 source URLs. 
    Do not invent facts beyond what the findings contain.""",
    response_format=FinalReport,
    middleware=[
        RestateMiddleware(),
        SummarizationMiddleware(
            model=init_durable_model("gpt-5.4-mini"),
            trigger=("tokens", 4000),
            keep=("messages", 10),
        ),
    ],
)


deep_research_agent = restate.VirtualObject("DeepResearchAgent")


@deep_research_agent.handler()
async def research(rst: restate.ObjectContext, query: str) -> FinalReport | None:
    history = await rst.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append(HumanMessage(id=str(rst.uuid()), content=query))

    # Stage 1 — plan, then wait for a human to approve it in Slack.
    result = await planner.ainvoke({"messages": history.messages})
    plan: ResearchPlan = result["structured_response"]

    # The /slack/interactivity webhook resolves this awakeable on click.
    awk_id, decision_future = rst.awakeable(type_hint=PlanDecision)
    await rst.run_typed("post-plan", post_plan, channel=rst.key(), plan=plan, awk_id=awk_id)
    decision: PlanDecision = await decision_future

    # Rejected: remember the proposed plan and bail. Human's next message gives feedback.
    if not decision.approved:
        msg = f"Proposed plan (rejected — revise per feedback):\n{plan.model_dump_json()}"
        history.messages.append(AIMessage(id=str(rst.uuid()), content=msg))
        rst.set("messages", history)
        return None

    # Stage 2 — fan out one ResearchAgent per subtopic, in parallel
    handles = [rst.service_call(investigate, arg=sub) for sub in plan.subtopics]
    await restate.gather(*handles)
    sub_reports = [await h for h in handles]

    # Stage 3 — synthesize the final report
    result = await writer.ainvoke({"messages": to_brief(query, plan, sub_reports)})
    report: FinalReport = result["structured_response"]

    # Stage 4 — deliver the rich report card back to the channel
    await rst.run_typed("slack-reply", post_report, topic=query, channel=rst.key(), report=report)

    history.messages.append(AIMessage(content=report.model_dump_json(), id=str(rst.uuid())))
    rst.set("messages", history)
    return report



# ----------- Autonomous Daily Research ---------------------


news_scout = create_agent(
    model="openai:gpt-5",
    tools=[web_search, extract_urls],
    system_prompt="""You are a news scout. Given a topic, use `web_search` with
    `time_range='day'` to find what's new in the last 24 hours. If a
    story looks important or unclear, follow up with `extract_urls` to
    read it in full. Keep the loop tight — at most 3 rounds of tool calls.
    Return a NewsDigest: a one-paragraph overview plus 3-5 distinct, concise news
    items (headline, 1-2 sentence summary, source URL).""",
    response_format=NewsDigest,
    middleware=[RestateMiddleware()],
)


@deep_research_agent.handler()
async def scan_news(rst: restate.ObjectContext, topic: str):
    # News scan agent
    result = await news_scout.ainvoke({"messages": f"Topic: {topic}"})
    news: NewsDigest = result["structured_response"]

    # Post to Slack
    await rst.run_typed("slack-news-update", post_news, topic=topic, channel=rst.key(), digest=news)

    # Update VO state to answer questions later
    history = await rst.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append(AIMessage(content=news.model_dump_json(), id=str(rst.uuid())))
    rst.set("messages", history)

    # Self-schedule tomorrow (same topic + channel)
    rst.object_send(scan_news, key=rst.key(), arg=topic, send_delay=timedelta(days=1))