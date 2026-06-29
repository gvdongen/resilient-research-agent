"""Slack delivery + prompt formatters. Each phase defines its own Tavily
@tool functions locally so the phase files stay readable on their own."""

import logging
import os
import random
from typing import Literal

from litellm import acompletion
from tavily import TavilyClient, BadRequestError
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from .schemas import NewsDigest, Plan, SubReport, Report, LLMRequest
from .stubs import stub_provider

Range = Literal["day", "week", "month", "year"]

RESTATE_HOST = os.environ.get("RESTATE_CLOUD_INGRESS") or "http://localhost:8080"

# ----------- Tavily Tools ---------------------

# Offline demo mode (see deep_research.OFFLINE): return canned results, no network/key needed.
OFFLINE = os.environ.get("OFFLINE") == "1"

FAILURE_PROBABILITY = 0

tavily_client = None if OFFLINE else TavilyClient()


def _stub_result(query: str) -> dict:
    return {
        "results": [
            {
                "title": f"[stub] Result for {query}",
                "url": "https://example.com",
                "content": f"Offline canned search content for: {query}",
            }
        ]
    }


def call_websearch_api(query: str, range: Range) -> dict:
    if random.random() < FAILURE_PROBABILITY:
        import time

        time.sleep(random.uniform(0, 2))
        raise Exception("Tavily API down")
    if OFFLINE:
        return {"query": query, "result": _stub_result(query)}
    try:
        result = tavily_client.search(
            query=query, time_range=range, search_depth="advanced"
        )
    except BadRequestError as e:
        # Non-transient: malformed request so propagate back to LLM
        result = f"BadRequestError: {str(e)}\n\nTry a different query."
    return {"query": query, "result": result}


async def call_extract_api(urls: list[str]) -> dict:
    if random.random() < FAILURE_PROBABILITY:
        raise Exception("Tavily API down")
    if OFFLINE:
        return _stub_result(", ".join(urls))
    try:
        return tavily_client.extract(urls=urls, extract_depth="advanced")
    except BadRequestError as e:
        # Non-transient: malformed request so propagate back to LLM
        return {"result": f"BadRequestError: {str(e)}\n\nTry a different query."}


def call_crawl_api(url: str, instructions: str = "") -> dict:
    if OFFLINE:
        return {"url": url, "result": _stub_result(url)}
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


def _slack_post(client: WebClient | None, **kwargs) -> str | None:
    """Try chat.postMessage; on auth/other Slack failure, treat as no client.

    Returns the message ts on success, None when the caller should fall back to
    the log-only path (no token, or token rejected by Slack)."""
    if client is None:
        return None
    try:
        return client.chat_postMessage(**kwargs)["ts"]
    except SlackApiError as e:
        err = e.response.get("error", "unknown") if e.response else "unknown"
        logger.warning("Slack post failed (%s) — falling back to log output.", err)
        return None


def post_update(channel: str, text: str) -> str:
    """Post a plain message to a Slack channel. Returns the message ts."""
    ts = _slack_post(_slack_client(), channel=channel, text=text)
    if ts is None:
        logger.info("\n=== Slack reply (channel=%s) ===\n%s\n", channel, text)
        return "log:reply"
    return ts


def to_brief(plan: Plan, sub_reports: list[SubReport]) -> str:
    return (
        f"# Topic\n{plan.topic}\n\n"
        f"# Plan rationale\n{plan.rationale}\n\n"
        "# Researcher findings\n\n"
        + "\n\n".join(
            f"## {sr.subtopic}\n{sr.findings}\n\nSources: {', '.join(sr.sources)}"
            for sr in sub_reports
        )
    )


def post_plan(channel: str, plan: Plan, awk_id: str) -> str:
    """Post a proposed research plan with Approve / Reject buttons for human review.

    The buttons carry `awk_id`, the Restate awakeable the orchestrator is parked
    on: Approve resolves it as approved so research proceeds; Reject resolves it
    as rejected, after which the human types their feedback as a normal channel
    message to trigger a revised plan. Returns ts."""
    subtopics_md = "\n".join(f"• {s}" for s in plan.subtopics)
    text = f"_{plan.rationale}_\n\n*Subtopics:*\n{subtopics_md}"

    def log_only() -> str:
        resolve_url = f"{RESTATE_HOST}/restate/awakeables/{awk_id}/resolve"
        auth = (
            ""
            if "localhost" in RESTATE_HOST
            else '-H "Authorization: Bearer $RESTATE_AUTH_TOKEN"'
        )
        logger.info(
            "\n=== Research plan (channel=%s) — needs approval ===\n%s\n\n"
            "▶ Approve:  curl %s %s --json '{\"approved\": true}'\n"
            "▶ Reject:   curl %s %s --json '{\"approved\": false}'  (then send feedback as a message)\n",
            channel,
            text,
            resolve_url,
            auth,
            resolve_url,
            auth,
        )
        return "log:plan"

    client = _slack_client()
    if client is None:
        return log_only()

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "📋 Research plan — needs your approval",
            },
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
    ts = _slack_post(
        client, channel=channel, text="Research plan needs your approval", blocks=blocks
    )
    return ts if ts is not None else log_only()


def post_news(topic: str, channel: str, digest: NewsDigest) -> str:
    """Post today's news digest. To dive deeper, the human just replies in the
    channel — that message triggers a research run. Returns the message ts."""
    items_md = "\n\n".join(
        f"*{idx + 1}. {i.headline}*\n{i.summary}\n<{i.url}>"
        for idx, i in enumerate(digest.items)
    )

    def log_only() -> str:
        logger.info(
            "\n=== Today's news: %s ===\n%s\n\n%s\n\n"
            "▶ Want to dive deeper? Just reply in the channel with what to research.\n",
            topic,
            digest.overview,
            items_md,
        )
        return "log:news"

    client = _slack_client()
    if client is None:
        return log_only()

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
    ts = _slack_post(
        client,
        channel=channel,
        text=f"Today's news: {topic}",
        blocks=blocks,
    )
    return ts if ts is not None else log_only()


def post_report(
    channel: str, report: Report, thread_ts: str | None = None
) -> str:
    """Post the FinalReport (rich Block Kit). Returns the message ts.

    If `thread_ts` is given, posts as a reply on that news card's thread."""

    def log_only() -> str:
        sections = "\n\n".join(f"## {s.heading}\n{s.body}" for s in report.sections)
        sources = "\n".join(f"• {s}" for s in report.sources)
        logger.info(
            "\n=== Deep research: %s ===\n# %s\n\n%s\n\n%s\n\nSources:\n%s\n",
            report.headline,
            report.executive_summary,
            sections,
            sources,
        )
        return "log:report"

    client = _slack_client()
    if client is None:
        return log_only()

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Deep research: {report.headline}"},
        },
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

    ts = _slack_post(
        client,
        channel=channel,
        text=report.headline,
        blocks=blocks,
        thread_ts=thread_ts,
    )
    return ts if ts is not None else log_only()


async def provider_call(req: LLMRequest) -> dict:
    if OFFLINE:
        return await stub_provider(req)  # canned responses live in utils/stubs.py
    response = await acompletion(
        model=req.model,
        messages=req.messages,
        tools=req.tools,
        response_format=req.output,
    )
    # exclude_none so the assistant message is clean to append back to the history
    return response.model_dump(exclude_none=True)


def to_schema(output_model) -> dict:
    """OpenAI strict json_schema format from a pydantic model. Our models use
    `extra="forbid"` and have no defaults, which strict json_schema requires."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": output_model.__name__,
            "schema": output_model.model_json_schema(),
            "strict": True,
        },
    }