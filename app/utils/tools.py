"""Slack delivery + prompt formatters. Each phase defines its own Tavily
@tool functions locally so the phase files stay readable on their own."""

import logging
import os
from typing import Literal
from tavily import TavilyClient, BadRequestError
from slack_sdk import WebClient
from .schemas import NewsDigest, ResearchPlan, Report, FinalReport

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


# ---- Slack tools ---------------------------------------------

logger = logging.getLogger("deep_research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def _slack_client() -> WebClient | None:
    """Return a WebClient if SLACK_BOT_TOKEN is set, else None."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    return WebClient(token=token) if token else None


def post_to_channel(channel: str, text: str) -> str:
    """Post a plain message to a Slack channel. Returns the message ts."""
    client = _slack_client()
    if client is None:
        logger.info("\n=== Slack reply (channel=%s) ===\n%s\n", channel, text)
        return "log:reply"
    resp = client.chat_postMessage(channel=channel, text=text)
    return resp["ts"]


def to_brief(topic: str, plan: ResearchPlan, sub_reports: list[Report]) -> str:
    return (
        f"# Topic\n{topic}\n\n"
        f"# Plan rationale\n{plan.rationale}\n\n"
        "# Researcher findings\n\n"
        + "\n\n".join(
            f"## {sr.subtopic}\n{sr.findings}\n\nSources: {', '.join(sr.sources)}"
            for sr in sub_reports
        )
    )


def post_plan(channel: str, plan: ResearchPlan, awk_id: str) -> str:
    """Post a proposed research plan with Approve / Reject buttons for human review.

    The buttons carry `awk_id`, the Restate awakeable the orchestrator is parked
    on: Approve resolves it as approved so research proceeds; Reject resolves it
    as rejected, after which the human types their feedback as a normal channel
    message to trigger a revised plan. Returns ts."""
    subtopics_md = "\n".join(f"• {s}" for s in plan.subtopics)
    text = f"_{plan.rationale}_\n\n*Subtopics:*\n{subtopics_md}"

    client = _slack_client()
    if client is None:
        resolve_url = f"{RESTATE_HOST}/restate/awakeables/{awk_id}/resolve"
        auth = "" if "localhost" in RESTATE_HOST else "-H \"Authorization: Bearer $RESTATE_AUTH_TOKEN\""
        logger.info(
            "\n=== Research plan (channel=%s) — needs approval ===\n%s\n\n"
            "▶ Approve:  curl %s %s --json '{\"approved\": true}'\n"
            "▶ Reject:   curl %s %s --json '{\"approved\": false}'  (then send feedback as a message)\n",
            channel, text, resolve_url, auth, resolve_url, auth,
        )
        return "log:plan"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "📋 Research plan — needs your approval"},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "style": "primary",
                    "text": {"type": "plain_text", "text": "✅ Approve"},
                    "action_id": f"plan_approve:{awk_id}",
                    "value": "approve",
                },
                {
                    "type": "button",
                    "style": "danger",
                    "text": {"type": "plain_text", "text": "✏️ Reject"},
                    "action_id": f"plan_reject:{awk_id}",
                    "value": "reject",
                },
            ],
        },
    ]
    resp = client.chat_postMessage(
        channel=channel, text="Research plan needs your approval", blocks=blocks
    )
    return resp["ts"]


def post_news(topic: str, channel: str, digest: NewsDigest) -> str:
    """Post today's news digest. To dive deeper, the human just replies in the
    channel — that message triggers a research run. Returns the message ts."""
    items_md = "\n\n".join(
        f"*{idx + 1}. {i.headline}*\n{i.summary}\n<{i.url}>"
        for idx, i in enumerate(digest.items)
    )

    client = _slack_client()
    if client is None:
        logger.info(
            "\n=== Today's news: %s ===\n%s\n\n%s\n\n"
            "▶ Want to dive deeper? Just reply in the channel with what to research.\n",
            topic,
            digest.overview,
            items_md,
        )
        return "log:news"

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
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "▶ *Want to dive deeper?* Reply in the channel with what to research.",
                }
            ],
        },
    ]
    resp = client.chat_postMessage(
        channel=channel,
        text=f"Today's news: {topic}",
        blocks=blocks,
    )
    return resp["ts"]


def post_report(
    topic: str, channel: str, report: FinalReport, thread_ts: str | None = None
) -> str:
    """Post the FinalReport (rich Block Kit). Returns the message ts.

    If `thread_ts` is given, posts as a reply on that news card's thread."""
    client = _slack_client()
    if client is None:
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
        thread_ts=thread_ts,
    )
    return resp["ts"]
