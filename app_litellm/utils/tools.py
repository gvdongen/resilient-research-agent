"""Tavily tools + the provider call + Slack delivery.

Slack is intentionally minimal: one `post_to_slack(channel, text)` that posts
markdown — or just logs it when there's no SLACK_BOT_TOKEN (the demo's default).
The session object formats each kind of update and calls it; nothing else writes
to Slack."""

import logging
import os
import random
from typing import Literal, TypeVar

from litellm import acompletion
from litellm.utils import function_to_dict
from restate import RestateDurableFuture
from restate.server_context import ServerDurableFuture
from tavily import TavilyClient, BadRequestError
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from .schemas import LLMRequest, Plan
from .stubs import stub_provider

T = TypeVar("T")

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


# ---- Slack delivery ----------------------------------------------------------

logger = logging.getLogger("deep_research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def format_plan(plan: dict) -> str:
    subtopics = "\n".join(f"• {s}" for s in plan["subtopics"])
    return f"*📋 Research plan — needs approval*\n_{plan['rationale']}_\n\n*Subtopics:*\n{subtopics}"


def format_report(report: dict) -> str:
    sections = "\n\n".join(f"*{s['heading']}*\n{s['body']}" for s in report["sections"])
    sources = "\n".join(f"• {s}" for s in report["sources"])
    return (
        f"*🔎 {report['headline']}*\n{report['executive_summary']}\n\n"
        f"{sections}\n\n*Sources:*\n{sources}"
    )

def _slack_client() -> WebClient | None:
    """Return a WebClient if SLACK_BOT_TOKEN is set, else None."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    return WebClient(token=token) if token else None


def post_to_slack(channel: str, text: str, awk_id: str | None = None) -> str:
    """Post markdown to a Slack channel — or just log it when there's no
    SLACK_BOT_TOKEN (the demo's default). Returns the message ts (or "log").

    This is the ONLY path to Slack: the session object formats each kind of
    update (plan / status / report) into markdown and calls this. Pass `awk_id`
    for a plan that needs approval — in Slack that adds Approve/Reject buttons
    (resolved by deploy/modal_app.py's /slack/interactivity), and in log mode it
    prints the resolve curl instead."""
    client = _slack_client()
    if client is None:
        if awk_id:
            text = f"{text}\n\n{_approval_curl(awk_id)}"
        logger.info("\n=== Slack (channel=%s) ===\n%s\n", channel, text)
        return "log"
    try:
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]
        if awk_id:
            blocks.append(_approval_buttons(awk_id))
        return client.chat_postMessage(channel=channel, text=text[:150], blocks=blocks)["ts"]
    except SlackApiError as e:
        err = e.response.get("error", "unknown") if e.response else "unknown"
        logger.warning("Slack post failed (%s) — logging instead.\n%s", err, text)
        return "log"


def _approval_buttons(awk_id: str) -> dict:
    """Approve/Reject actions row. The action_ids carry the awakeable id; the
    button clicks are resolved by deploy/modal_app.py's /slack/interactivity."""
    return {
        "type": "actions",
        "elements": [
            {"type": "button", "style": "primary", "value": "approve",
             "text": {"type": "plain_text", "text": "✅ Approve"},
             "action_id": f"plan_approve:{awk_id}"},
            {"type": "button", "style": "danger", "value": "reject",
             "text": {"type": "plain_text", "text": "✏️ Reject"},
             "action_id": f"plan_reject:{awk_id}"},
        ],
    }


def _approval_curl(awk_id: str) -> str:
    """Copy-pasteable curl to approve/reject a plan parked on an awakeable —
    the log-mode fallback when there are no Slack buttons to click."""
    url = f"{RESTATE_HOST}/restate/awakeables/{awk_id}/resolve"
    auth = "" if "localhost" in RESTATE_HOST else '-H "Authorization: Bearer $RESTATE_AUTH_TOKEN"'
    return (
        f"▶ Approve:  curl {url} {auth} --json '{{\"approved\": true}}'\n"
        f"▶ Reject:   curl {url} {auth} --json '{{\"approved\": false}}'"
    )


def to_brief(plan: dict, sub_reports: list[dict]) -> str:
    return (
        f"# Topic\n{plan['topic']}\n\n"
        f"# Plan rationale\n{plan['rationale']}\n\n"
        "# Researcher findings\n\n"
        + "\n\n".join(
            f"## {sr['subtopic']}\n{sr['findings']}\n\nSources: {', '.join(sr['sources'])}"
            for sr in sub_reports
        )
    )


# ---- Provider call + tool helper --------------------------------------------


async def provider_call(req: LLMRequest) -> dict:
    if OFFLINE:
        return await stub_provider(req)  # canned responses live in utils/stubs.py
    response = await acompletion(
        model=req.model,
        messages=req.msgs,
        tools=req.tools,
        response_format=req.output_schema,
    )
    # exclude_none so the assistant message is clean to append back to the history
    return response.model_dump(exclude_none=True)


def to_tool(fn) -> dict:
    """Wrap a plain async function as an OpenAI-style tool definition, deriving
    name/description/parameters from its signature and docstring."""
    return {"type": "function", "function": function_to_dict(fn)}


async def peek(fut: RestateDurableFuture[T]) -> T | None:
    assert isinstance(fut, ServerDurableFuture)
    if fut.is_completed():
        return await fut
    return None


def append(history: ChatHistory, text: str) -> None:
    history.messages.append({"role": "user", "content": f"Steering update from the user — incorporate this:\n{text}"})