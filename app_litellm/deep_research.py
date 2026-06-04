"""
Deep Research agent — custom litellm tool loop (no framework).

Same shape as `app/deep_research.py`, but the agent loop is hand-written
against `litellm.acompletion`:
- gets triggered by messages in a Slack channel
- kicks off planner, parallel research agents, and writer
- posts reply back in Slack channel

Option to daily scan news on a topic.
"""

import json
from datetime import timedelta

import restate
from litellm import acompletion
from litellm.types.utils import ModelResponse
from litellm.utils import function_to_dict
from pydantic import BaseModel

from utils.schemas import (
    ChatHistory,
    FinalReport,
    NewsDigest,
    PlanDecision,
    Report,
    ResearchPlan,
)
from utils.tools import (
    Range,
    post_news,
    post_plan,
    post_report,
    tavily_crawl,
    tavily_extract,
    tavily_search,
    to_brief,
)

# ----------- Tools (raw Tavily wrappers; dispatched via ctx.run_typed) ----------


async def web_search(query: str, time_range: Range = "month") -> dict:
    """Search the web. time_range is one of: day, week, month, year."""
    return tavily_search(query=query, range=time_range)


async def extract_urls(urls: list[str]) -> dict:
    """Return the full readable text of a list of web pages."""
    return await tavily_extract(urls=urls)


async def crawl_site(url: str, instructions: str = "") -> dict:
    """Crawl a website (guided by natural-language instructions) and return content from up to 10 pages."""
    return tavily_crawl(url=url, instructions=instructions)


TOOL_SPECS = [
    {"type": "function", "function": function_to_dict(fn)}
    for fn in (web_search, extract_urls, crawl_site)
]

TOOLS = {
    "web_search": web_search,
    "extract_urls": extract_urls,
    "crawl_site": crawl_site,
}


# ----------- Generic durable agent loop ---------------------


async def call_llm(
    messages: list[dict],
    output_model: type[BaseModel],
    tools: list | None = None,
) -> ModelResponse:
    kwargs: dict = {
        "model": "gpt-5",
        "messages": messages,
        "response_format": output_model,
        "tools": tools,
    }
    return await acompletion(**kwargs)


async def run_agent(
    ctx: restate.Context,
    messages: list[dict],
    *,
    output_model: type[BaseModel],
    tools: list | None = None,
    max_turns: int = 3,
) -> BaseModel:
    """Bounded tool-using agent loop. Every LLM call + tool call is journaled,
    so retries replay from the journal instead of re-executing paid calls."""
    msgs = list(messages)
    for _ in range(max_turns):
        response = await ctx.run_typed(
            "llm",
            call_llm,
            messages=msgs,
            output_model=output_model,
            tools=tools,
        )
        msg = response.choices[0].message
        msgs.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return output_model.model_validate_json(msg.content)

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
            msgs.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(await h)}
            )

    # Budget exhausted — force a final structured answer with no more tools
    msgs.append(
        {
            "role": "user",
            "content": "Turn budget exhausted. Return the result now with what you have.",
        }
    )
    response = await ctx.run_typed(
        "llm-final",
        call_llm,
        messages=msgs,
        output_model=output_model,
        tools=None,
    )
    return output_model.model_validate_json(response.choices[0].message.content)


# ----------- System prompts ---------------------

RESEARCHER_SYSTEM = """You have web_search, extract_urls,
and crawl_site available. Investigate the assigned subtopic thoroughly:
search 3-5 topics with recency-appropriate time_range, then read the most
promising sources in full. Keep the loop tight — at most 2 rounds of
tool calls. Cite every claim with a URL. Stop as soon as you have
enough to write a tight 200-400 word findings section."""

PLANNER_SYSTEM = """You are a senior research planner. Given a topic and a digest of
today's news on it, produce a tight research plan: a short rationale
(what's worth digging into and why) plus 3-5 sharply scoped subtopics.
Each subtopic should be a self-contained research question that a
separate researcher can investigate in parallel without overlap with
the others. Prefer subtopics that dig into the most consequential
items from today's news."""

WRITER_SYSTEM = """You are a senior editor turning raw research notes into a polished
report. Take the topic, the plan's rationale, and the per-subtopic
findings. Write a concise report about 3-5 key findings (at most 100 words explanation).
Add a de-duplicated list of the top 5 source URLs. 
Do not invent facts beyond what the findings contain."""

NEWS_SCOUT_SYSTEM = """You are a news scout. Given a topic, use `web_search` with
`time_range='day'` to find what's new in the last 24 hours. If a
story looks important or unclear, follow up with `extract_urls` to
read it in full. Keep the loop tight — at most 3 rounds of tool calls.
Return a NewsDigest: a one-paragraph overview plus 3-5 distinct, concise news
items (headline, 1-2 sentence summary, source URL)."""


# ----------- Durable Agents ---------------------

research_agent = restate.Service("ResearchAgent")


@research_agent.handler()
async def investigate(rst: restate.Context, topic: str) -> Report:
    return await run_agent(
        rst,
        messages=[
            {"role": "system", "content": RESEARCHER_SYSTEM},
            {"role": "user", "content": f"Topic: {topic}"},
        ],
        output_model=Report,
        tools=TOOL_SPECS,
    )


# ----------- Durable Agentic Workflows ---------------------


async def deep_research(
    rst: restate.ObjectContext, query: str, history: ChatHistory
) -> dict:
    # Stage 1 — plan (sees the full chat history including any prior rejected plans)
    plan: ResearchPlan = await run_agent(
        rst,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            *history.messages,
        ],
        output_model=ResearchPlan,
    )

    # Stage 2 — human approval of plan
    awk_id, decision_promise = rst.awakeable(type_hint=PlanDecision)
    await rst.run_typed(
        "post-plan", post_plan, channel=rst.key(), plan=plan, awk_id=awk_id
    )
    decision: PlanDecision = await decision_promise

    # Rejected — return the proposed plan and wait for feedback as a new turn
    if not decision.approved:
        msg = (
            f"Proposed plan (rejected — revise per feedback):\n{plan.model_dump_json()}"
        )
        return {"role": "assistant", "content": msg}

    # Stage 3 — fan out one ResearchAgent per subtopic, in parallel
    handles = [rst.service_call(investigate, arg=sub) for sub in plan.subtopics]
    await restate.gather(*handles)
    sub_reports = [await h for h in handles]

    # Stage 4 — synthesize the final report
    report: FinalReport = await run_agent(
        rst,
        messages=[
            {"role": "system", "content": WRITER_SYSTEM},
            {"role": "user", "content": to_brief(query, plan, sub_reports)},
        ],
        output_model=FinalReport,
    )

    # Stage 5 — deliver the rich report card back to the channel
    await rst.run_typed(
        "post-report", post_report, topic=query, channel=rst.key(), report=report
    )

    return {"role": "assistant", "content": report.model_dump_json()}


# ----------- Durable Sessions ---------------------

deep_research_agent = restate.VirtualObject("DeepResearchAgent")


@deep_research_agent.handler()
async def research(rst: restate.ObjectContext, query: str) -> FinalReport | None:
    history = await rst.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append({"role": "user", "content": query})

    response = await deep_research(rst, query, history)

    history.messages.append(response)
    rst.set("messages", history)

    return response


# ----------- Autonomous Research ---------------------


@deep_research_agent.handler()
async def scan_news(rst: restate.ObjectContext, topic: str):
    # News scan agent
    news: NewsDigest = await run_agent(
        rst,
        messages=[
            {"role": "system", "content": NEWS_SCOUT_SYSTEM},
            {"role": "user", "content": f"Topic: {topic}"},
        ],
        output_model=NewsDigest,
        tools=TOOL_SPECS,
    )

    # Post to Slack
    await rst.run_typed(
        "post-news", post_news, topic=topic, channel=rst.key(), digest=news
    )

    # Update VO state so the planner can pick up the news on a follow-up "research" call
    history = await rst.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append({"role": "assistant", "content": news.model_dump_json()})
    rst.set("messages", history)

    # Self-schedule tomorrow (same topic + channel)
    rst.object_send(scan_news, key=rst.key(), arg=topic, send_delay=timedelta(days=1))
