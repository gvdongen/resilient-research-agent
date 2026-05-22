"""Slack delivery + prompt formatters. Each phase defines its own Tavily
@tool functions locally so the phase files stay readable on their own."""

import logging
import os
from typing import Literal

from tavily import TavilyClient, BadRequestError
from slack_sdk import WebClient

from .schemas import NewsDigest, ResearchPlan, SubReport, FinalReport

Range = Literal["day", "week", "month", "year"]

RESTATE_HOST = os.environ.get("RESTATE_CLOUD_INGRESS") or "http://localhost:8080"

# ----------- Tavily Tools ---------------------

tavily_client = TavilyClient()


def tavily_search(query: str, range: Range) -> dict:
    try:
        result = tavily_client.search(query=query, time_range=range, search_depth="advanced")
    except BadRequestError as e:
        # Non-transient: malformed request so propagate back to LLM
        result = f"BadRequestError: {str(e)}\n\nTry a different query."
    return {"query": query, "result": result}


def tavily_extract(urls: list[str]) -> dict:
    try:
        return tavily_client.extract(urls=urls, extract_depth="advanced")
    except BadRequestError as e:
        # Non-transient: malformed request so propagate back to LLM
        return {"result": f"BadRequestError: {str(e)}\n\nTry a different query."}


def tavily_crawl(url: str, instructions: str = "") -> dict:
    try:
        result = tavily_client.crawl(url=url, instructions=instructions)
    except BadRequestError as e:
        # Non-transient: malformed request so propagate back to LLM
        result = f"BadRequestError: {str(e)}\n\nTry a different query."
    return {"url": url, "result": result}


# ---- Middleware state isolation ---------------------------------------------

logger = logging.getLogger("deep_research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def _slack() -> tuple[WebClient, str] | None:
    """Return (client, channel) if both env vars are set, else None."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_DIGEST_CHANNEL_ID")
    if not token or not channel:
        return None
    return WebClient(token=token), channel


def summarize(topic: str, news: NewsDigest) -> str:
    news_lines = "\n".join(
        f"- {it.headline}: {it.summary} ({it.url})" for it in news.items
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


def post_news(topic: str, digest: NewsDigest, awk_id: str) -> str:
    """Post today's news + curl to trigger the deep-dive. Returns ts."""
    resolve_url = f"{RESTATE_HOST}/restate/awakeables/{awk_id}/resolve"
    auth_header = "" if "localhost" in RESTATE_HOST else "-H \"Authorization: Bearer $RESTATE_AUTH_TOKEN\""
    yes_cmd = f"curl {resolve_url} {auth_header} --json '\"Tell me more about the first story\"'"

    items_md = "\n\n".join(
        f"*{i.headline}*\n{i.summary}\n<{i.url}>" for i in digest.items
    )

    slack = _slack()
    if slack is None:
        logger.info(
            "\n=== Today's news: %s ===\n%s\n\n%s\n\n▶ Want to dive deeper?\n%s\n",
            topic,
            digest.overview,
            items_md,
            yes_cmd,
        )
        return "log:news"

    client, channel = slack
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Today's news: {topic}"},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": f"_{digest.overview}_"}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": items_md}},
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ("▶ *Want to dive deeper?*\n\n" f"```{yes_cmd}```\n\n"),
            },
        },
    ]
    resp = client.chat_postMessage(
        channel=channel,
        text=f"Today's news: {topic} — dive deeper?",
        blocks=blocks,
    )
    return resp["ts"]


def post_report(topic: str, report: FinalReport) -> str:
    """Post the FinalReport (rich Block Kit). Returns the message ts."""
    slack = _slack()
    if slack is None:
        sections = "\n\n".join(f"## {s.heading}\n{s.body}" for s in report.sections)
        sources = "\n".join(f"• {s}" for s in report.sources)
        logger.info(
            "\n=== Deep research: %s ===\n# %s\n\n%s\n\n%s\n\nSources:\n%s\n",
            topic,
            report.headline,
            report.executive_summary,
            sections,
            sources,
        )
        return "log:report"

    client, channel = slack
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Deep research: {topic}"},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{report.headline}*"}},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": report.executive_summary},
        },
        {"type": "divider"},
    ]
    for sec in report.sections:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{sec.heading}*\n{sec.body}"},
            }
        )
    if report.sources:
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Sources:\n"
                        + "\n".join(f"• <{s}>" for s in report.sources),
                    }
                ],
            }
        )

    resp = client.chat_postMessage(
        channel=channel,
        text=report.headline,
        blocks=blocks,
    )
    return resp["ts"]
