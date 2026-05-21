"""Phase 3 — Autonomous daily deep-research loop (custom litellm loop).

`DeepResearchAgent.run(topic)`:
  1. NewsScoutAgent scans the news on a topic and writes a digest.
  2. Post the digest to Slack with a curl to resolve an awakeable to request
     a deep dive on a news topic. The orchestrator suspends on `restate.select`
     until the awakeable fires or a 24h timeout elapses.
  3. If a deep-dive topic arrives →
     PlannerAgent → N parallel ResearchAgents → WriterAgent
     and then the FinalReport is posted to Slack.
  4. Self-schedule tomorrow's run via `ctx.service_send(..., send_delay=days=1)`.

Self-contained: defines its own Tavily tools, its own `run_agent` helper, and
its own researcher (independent from Phase 1).
"""

import json
from datetime import timedelta
from typing import Literal

import restate
from litellm import acompletion
from litellm.types.utils import ModelResponse
from litellm.utils import function_to_dict
from pydantic import BaseModel
from tavily import TavilyClient

from utils.schemas import (
    DailyResult,
    FinalReport,
    NewsDigest,
    ResearchPlan,
    ResearchRequest,
    SubReport,
)
from utils.tools import post_news, post_report, summarize, to_brief

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


# ----------- Generic agent loop ---------------------


async def call_llm(
    messages: list,
    output_model: type[BaseModel],
    tools: list | None = TOOL_SPECS,
) -> ModelResponse:
    kwargs: dict = {
        "model": "gpt-5",
        "messages": messages,
        "response_format": output_model,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return await acompletion(**kwargs)


async def run_agent(
    ctx: restate.Context,
    system: str,
    user: str,
    output_model: type[BaseModel],
    use_tools: bool = True,
    max_turns: int = 3,
) -> BaseModel:
    """Bounded tool-using agent loop, journaling every LLM call and tool call."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if not use_tools:
        response = await ctx.run_typed(
            "llm",
            call_llm,
            messages=messages,
            output_model=output_model,
            tools=None,
        )
        return output_model.model_validate_json(response.choices[0].message.content)

    for _ in range(max_turns):
        response = await ctx.run_typed(
            "llm",
            call_llm,
            messages=messages,
            output_model=output_model,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return output_model.model_validate_json(msg.content)

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

    messages.append(
        {
            "role": "user",
            "content": "You've used your turn budget. Write the result now with what you have.",
        }
    )
    response = await ctx.run_typed(
        "llm-final",
        call_llm,
        messages=messages,
        output_model=output_model,
        tools=None,
    )
    return output_model.model_validate_json(response.choices[0].message.content)


# ----------- System prompts ---------------------

NEWS_SCOUT_SYSTEM = (
    "You are a news scout. Given a topic, use `web_search` with "
    "`time_range='day'` to find what's new in the last 24 hours. If a "
    "story looks important or unclear, follow up with `extract_urls` to "
    "read it in full. Return a NewsDigest: a one-paragraph overview plus "
    "3-8 distinct news items (headline, 1-2 sentence summary, source URL)."
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


# ----------- Stage services ---------------------

news_scout_agent = restate.Service("NewsScoutAgent")


@news_scout_agent.handler()
async def scan(ctx: restate.Context, topic: str) -> NewsDigest:
    return await run_agent(ctx, NEWS_SCOUT_SYSTEM, f"Topic: {topic}", NewsDigest)


planner_agent = restate.Service("PlannerAgent")


@planner_agent.handler()
async def plan(ctx: restate.Context, req: ResearchRequest) -> ResearchPlan:
    return await run_agent(
        ctx,
        PLANNER_SYSTEM,
        summarize(req.topic, req.news),
        ResearchPlan,
        use_tools=False,
    )


research_agent = restate.Service("ResearchAgent")


@research_agent.handler()
async def investigate(ctx: restate.Context, subtopic: str) -> SubReport:
    return await run_agent(ctx, RESEARCHER_SYSTEM, f"Subtopic: {subtopic}", SubReport)


writer_agent = restate.Service("WriterAgent")


@writer_agent.handler()
async def write(ctx: restate.Context, brief: str) -> FinalReport:
    return await run_agent(ctx, WRITER_SYSTEM, brief, FinalReport, use_tools=False)


# ----------- Orchestrator ---------------------

agent = restate.Service("DeepResearchAgent")


@agent.handler()
async def run(ctx: restate.Context, topic: str) -> DailyResult:
    # Self-schedule tomorrow
    ctx.service_send(run, arg=topic, send_delay=timedelta(days=1))

    # Stage 1 — daily news scan
    news = await ctx.service_call(scan, arg=topic)

    # Stage 2 — send digest to Slack, ask for a deep-dive topic via awakeable
    awk_id, decision_promise = ctx.awakeable(type_hint=str)
    await ctx.run_typed(
        "slack-news-update", post_news, topic=topic, digest=news, awk_id=awk_id
    )

    # Stage 3 — suspend up to 24h waiting for the user's reply
    match await restate.select(
        decision=decision_promise,
        timeout=ctx.sleep(timedelta(days=1)),
    ):
        case ["decision", deep_dive_topic]:
            pass
        case _:
            return DailyResult(news=news)

    # Stage 4 — plan
    research_plan = await ctx.service_call(
        plan, arg=ResearchRequest(topic=deep_dive_topic, news=news)
    )

    # Stage 5 — fan out one ResearchAgent per subtopic, in parallel
    handles = [
        ctx.service_call(investigate, arg=sub) for sub in research_plan.subtopics
    ]
    await restate.gather(*handles)
    sub_reports = [await h for h in handles]

    # Stage 6 — synthesize the final report
    report = await ctx.service_call(
        write, arg=to_brief(deep_dive_topic, research_plan, sub_reports)
    )

    # Stage 7 — deliver the report
    await ctx.run_typed(
        "slack-topic-report", post_report, topic=deep_dive_topic, report=report
    )
    return DailyResult(news=news, report=report)
