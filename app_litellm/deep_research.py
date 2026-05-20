"""DeepResearchAgent — autonomous daily loop, custom-agent-loop edition.

Each day:
  1. NewsScoutAgent scans the news on a topic and writes a digest.
  2. Post the digest to Slack with a curl to resolve an awakeable to request
     a deep dive on a news topic. The orchestrator suspends on `restate.select`
     until the awakeable fires or a 24h timeout elapses.
  3. If a deep-dive topic arrives →
     PlannerAgent → N parallel ResearchAgents → WriterAgent
     and then the FinalReport is posted to Slack.
  4. Self-schedule tomorrow's run via `ctx.service_send(..., send_delay=days=1)`.

The agent loop is hand-rolled (no LangChain): `run_agent` in utils/tools.py
journals each LLM turn and each Tavily tool call via `ctx.run_typed`.
"""

from datetime import timedelta

import restate

from app.utils.schemas import (
    DailyResult,
    FinalReport,
    NewsDigest,
    ResearchPlan,
    SubReport,
)
from app.utils.tools import (
    post_news,
    post_report,
    run_agent,
    summarize,
    to_brief,
)


# ----------- System prompts ---------------------

NEWS_SCOUT_SYSTEM = (
    "You are a news scout. Given a topic, use `web_search` with "
    "`time_range='day'` to find what's new in the last 24 hours. If a "
    "story looks important or unclear, follow up with `extract_urls` to "
    "read it in full. Return a NewsDigest: a one-paragraph overview plus "
    "3-8 distinct news items (headline, 1-2 sentence summary, source URL). "
    "Do not speculate beyond what's in the sources."
)

PLANNER_SYSTEM = (
    "You are a senior research planner. Given a topic and a digest of "
    "today's news on it, produce a tight research plan: a short rationale "
    "(what's worth digging into and why) plus 3-5 sharply scoped subtopics. "
    "Each subtopic should be a self-contained research question that a "
    "separate researcher can investigate in parallel without overlap with "
    "the others. Prefer subtopics that dig into the most consequential "
    "items from today's news."
)

RESEARCHER_SYSTEM = (
    "You are a focused research analyst. You have web_search, extract_urls, "
    "and crawl_site available. Investigate the assigned subtopic thoroughly: "
    "search with recency-appropriate time_range, then read the most "
    "promising sources in full. Cite every claim with a URL. Stop as soon as "
    "you have enough to write a tight 200-400 word findings section."
)

WRITER_SYSTEM = (
    "You are a senior editor turning raw research notes into a polished "
    "report. Take the topic, the plan's rationale, and the per-subtopic "
    "findings. Produce: a sharp headline, a 3-4 sentence executive summary, "
    "one section per subtopic (heading + 150-250 word body in markdown), "
    "and a de-duplicated list of source URLs. Cite inline where it adds "
    "credibility. Do not invent facts beyond what the findings contain."
)


# ----------- Agent services ---------------------

news_scout_agent = restate.Service("NewsScoutAgent")


@news_scout_agent.handler()
async def scan(ctx: restate.Context, topic: str) -> NewsDigest:
    return await run_agent(ctx, NEWS_SCOUT_SYSTEM, f"Topic: {topic}", NewsDigest)


planner_agent = restate.Service("PlannerAgent")


@planner_agent.handler()
async def plan(ctx: restate.Context, brief: str) -> ResearchPlan:
    return await run_agent(ctx, PLANNER_SYSTEM, brief, ResearchPlan, use_tools=False)


research_agent = restate.Service("ResearchAgent")


@research_agent.handler()
async def investigate(ctx: restate.Context, subtopic: str) -> SubReport:
    return await run_agent(ctx, RESEARCHER_SYSTEM, f"Subtopic: {subtopic}", SubReport)


writer_agent = restate.Service("WriterAgent")


@writer_agent.handler()
async def write(ctx: restate.Context, brief: str) -> FinalReport:
    return await run_agent(ctx, WRITER_SYSTEM, brief, FinalReport, use_tools=False)


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
    await ctx.run_typed(
        "slack-news-update", post_news, topic=topic, digest=news, awk_id=awk_id
    )

    match await restate.select(
        decision=decision_promise,
        timeout=ctx.sleep(timedelta(days=1)),
    ):
        case ["timeout"]:
            return DailyResult(news=news)
        case ["decision", answer]:
            deep_dive_topic = answer

    # Stage 3 — plan
    research_plan = await ctx.service_call(
        plan, arg=summarize(deep_dive_topic, news)
    )

    # Stage 4 — fan out one Researcher per subtopic, in parallel
    handles = [
        ctx.service_call(investigate, arg=sub)
        for sub in research_plan.subtopics
    ]
    await restate.gather(*handles)
    sub_reports = [await h for h in handles]

    # Stage 5 — synthesize the final report
    report = await ctx.service_call(
        write, arg=to_brief(deep_dive_topic, research_plan, sub_reports)
    )

    # Stage 6 — deliver the report
    await ctx.run_typed(
        "slack-topic-report", post_report, topic=deep_dive_topic, report=report
    )
    return DailyResult(news=news, report=report)
