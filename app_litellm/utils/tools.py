"""Shared helpers for the deep-research pipeline: Tavily tools, the LLM call,
the generic agent loop, prompt formatters, and Slack delivery."""

import json
import os
from typing import Literal

import restate
from litellm import acompletion
from litellm.types.utils import ModelResponse
from litellm.utils import function_to_dict
from pydantic import BaseModel
from slack_sdk import WebClient
from tavily import TavilyClient

from .schemas import FinalReport, NewsDigest, ResearchPlan, SubReport

Range = Literal["day", "week", "month", "year"]


# ---- Tavily-backed tools ----------------------------------------------------

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


# ---- LLM call ---------------------------------------------------------------

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


# ---- Generic agent loop -----------------------------------------------------

async def run_agent(
    ctx: restate.Context,
    system: str,
    user: str,
    output_model: type[BaseModel],
    use_tools: bool = True,
    max_turns: int = 2,
) -> BaseModel:
    """Run a custom tool-using agent loop, journaling every LLM call and every
    tool call. Returns a parsed `output_model` instance."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if not use_tools:
        response = await ctx.run_typed(
            "llm", call_llm, messages=messages, output_model=output_model, tools=None,
        )
        return output_model.model_validate_json(response.choices[0].message.content)

    for _ in range(max_turns):
        response = await ctx.run_typed(
            "llm", call_llm, messages=messages, output_model=output_model,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return output_model.model_validate_json(msg.content)

        handles = [
            ctx.run_typed(
                tc.function.name, TOOLS[tc.function.name],
                **json.loads(tc.function.arguments),
            )
            for tc in msg.tool_calls
        ]
        await restate.gather(*handles)
        for tc, h in zip(msg.tool_calls, handles):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(await h),
            })

    messages.append({
        "role": "user",
        "content": "You've used your turn budget. Write the result now with what you have.",
    })
    response = await ctx.run_typed(
        "llm-final", call_llm, messages=messages, output_model=output_model, tools=None,
    )
    return output_model.model_validate_json(response.choices[0].message.content)


# ---- Prompt formatters ------------------------------------------------------

def summarize(topic: str, news: NewsDigest) -> str:
    news_lines = "\n".join(
        f"- {it.headline}: {it.summary} ({it.url})"
        for it in news.items
    )
    return (
        f"Topic: {topic}\n\n"
        f"Today's news overview: {news.overview}\n\n"
        f"Today's news items:\n{news_lines}\n\n"
        "Produce a ResearchPlan."
    )


def to_brief(topic: str, plan: ResearchPlan, sub_reports: list[SubReport]) -> str:
    return (
        f"# Topic\n{topic}\n\n"
        f"# Plan rationale\n{plan.rationale}\n\n"
        "# Researcher findings\n\n"
        + "\n\n".join(
            f"## {sr.subtopic}\n{sr.findings}\n\nSources: {', '.join(sr.sources)}"
            for sr in sub_reports
        )
    )


# ---- Slack delivery ---------------------------------------------------------

slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])


def post_news(topic: str, digest: NewsDigest, awk_id: str) -> str:
    """Post today's news + curl to trigger the deep-dive. Returns ts."""
    resolve_url = f"http://localhost:8080/restate/awakeables/{awk_id}/resolve"
    yes_cmd = f"curl {resolve_url} --json '\"Tell me more about ...\"'"

    items_md = "\n\n".join(
        f"*{i.headline}*\n{i.summary}\n<{i.url}>"
        for i in digest.items
    )

    blocks: list[dict] = [
        {"type": "header",
         "text": {"type": "plain_text", "text": f"Today's news: {topic}"}},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": f"_{digest.overview}_"}},
        {"type": "divider"},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": items_md}},
        {"type": "divider"},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": (
             "▶ *Want to dive deeper?*\n\n"
             f"```{yes_cmd}```\n\n"
         )}},
    ]
    resp = slack_client.chat_postMessage(
        channel=os.environ["SLACK_DIGEST_CHANNEL_ID"],
        text=f"Today's news: {topic} — dive deeper?",
        blocks=blocks,
    )
    return resp["ts"]


def post_report(topic: str, report: FinalReport) -> str:
    """Post the FinalReport (rich Block Kit). Returns the message ts."""
    blocks: list[dict] = [
        {"type": "header",
         "text": {"type": "plain_text", "text": f"Deep research: {topic}"}},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": f"*{report.headline}*"}},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": report.executive_summary}},
        {"type": "divider"},
    ]
    for sec in report.sections:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f"*{sec.heading}*\n{sec.body}"},
        })
    if report.sources:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn",
                          "text": "Sources:\n" + "\n".join(f"• <{s}>" for s in report.sources)}],
        })

    resp = slack_client.chat_postMessage(
        channel=os.environ["SLACK_DIGEST_CHANNEL_ID"],
        text=report.headline,
        blocks=blocks,
    )
    return resp["ts"]
