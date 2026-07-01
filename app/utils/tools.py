"""Tavily tools + the provider call + Slack delivery.

Slack is intentionally minimal: one `post_to_slack(channel, text)` that posts
markdown — or just logs it when there's no SLACK_BOT_TOKEN (the demo's default).
The session object formats each kind of update and calls it; nothing else writes
to Slack."""

import json
import logging
import os
import random
from typing import Any, Literal, TypeVar

from litellm import acompletion
from litellm.utils import function_to_dict
from restate import RestateDurableFuture
from restate.server_context import ServerDurableFuture
from tavily import TavilyClient, BadRequestError
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from .config import FAST_MODEL
from .prompts import CLASSIFIER, PLANNER, WRITER
from .schemas import LLMRequest, Plan, ChatHistory, Strategy, Report
from .stubs import stub_provider

T = TypeVar("T")

Range = Literal["day", "week", "month", "year"]

RESTATE_HOST = os.environ.get("RESTATE_CLOUD_INGRESS") or "http://localhost:8080"

# Max characters of any single tool result that reaches the model. Tavily
# extract/crawl return full page text, which can be hundreds of thousands of
# tokens; clipping each source here keeps one tool message from blowing the
# context window (the SummarizationMiddleware handles cross-message growth).
MAX_TOOL_CHARS = int(os.environ.get("MAX_TOOL_CHARS", "4000"))


def cap(result: Any, limit: int = MAX_TOOL_CHARS) -> str:
    """Serialize a tool result and clip it to `limit` chars, keeping the head."""
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"

# ----------- Tavily Tools ---------------------

# Offline demo mode (see deep_research.OFFLINE): return canned results, no network/key needed.
OFFLINE = os.environ.get("OFFLINE") == "1"

FAILURE_PROBABILITY = 0.1

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


def call_websearch_api(query: str, range: Range) -> str:
    if random.random() < FAILURE_PROBABILITY:
        import time

        time.sleep(random.uniform(0, 2))
        raise Exception("Websearch API down")
    if OFFLINE:
        return cap({"query": query, "result": _stub_result(query)})
    try:
        result = tavily_client.search(
            query=query, time_range=range, search_depth="fast"
        )
    except BadRequestError as e:
        # Non-transient: malformed request so propagate back to LLM
        result = f"BadRequestError: {str(e)}\n\nTry a different query."
    return cap({"query": query, "result": result})


async def call_extract_api(urls: list[str]) -> str:
    if random.random() < FAILURE_PROBABILITY:
        raise Exception("Websearch API down")
    if OFFLINE:
        return cap(_stub_result(", ".join(urls)))
    try:
        return cap(tavily_client.extract(urls=urls, extract_depth="basic"))
    except BadRequestError as e:
        # Non-transient: malformed request so propagate back to LLM
        return cap({"result": f"BadRequestError: {str(e)}\n\nTry a different query."})


def call_crawl_api(url: str, instructions: str = "") -> str:
    if OFFLINE:
        return cap({"url": url, "result": _stub_result(url)})
    try:
        result = tavily_client.crawl(url=url, instructions=instructions, extract_depth="basic")
    except BadRequestError as e:
        # Non-transient: malformed request so propagate back to LLM
        result = f"BadRequestError: {str(e)}\n\nTry a different query."
    return cap({"url": url, "result": result})


# ---- Slack delivery ----------------------------------------------------------

logger = logging.getLogger("deep_research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def format_plan(plan: dict, awk_id: str) -> dict:
    subtopics = "\n".join(f"• {s}" for s in plan["subtopics"])
    msg = f"*📋 Research plan — needs approval*\n_{plan['rationale']}_\n\n*Subtopics:*\n{subtopics}"
    return {"text": msg, "awk_id": awk_id}


def format_report(report: dict, inv_id: str) -> dict:
    sections = "\n\n".join(f"*{s['heading']}*\n{s['body']}" for s in report["sections"])
    sources = "\n".join(f"• {s}" for s in report["sources"])
    msg = (
        f"*🔎 {report['headline']}*\n{report['executive_summary']}\n\n"
        f"{sections}\n\n*Sources:*\n{sources}"
    )
    return {"text": msg, "inv_id": inv_id}

def _slack_client() -> WebClient | None:
    """Return a WebClient if SLACK_BOT_TOKEN is set, else None."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    return WebClient(token=token) if token else None


# Slack rejects a section block whose text exceeds 3000 chars (`invalid_blocks`).
# The report easily overflows one block, so split it into several.
SLACK_SECTION_LIMIT = 3000


def _mrkdwn_sections(text: str, limit: int = SLACK_SECTION_LIMIT) -> list[dict]:
    """Split markdown into Slack section blocks, each <= limit chars, breaking on
    paragraph then line boundaries where possible so formatting stays intact."""
    blocks, remaining = [], text
    while remaining:
        if len(remaining) <= limit:
            chunk, remaining = remaining, ""
        else:
            window = remaining[:limit]
            cut = window.rfind("\n\n")
            if cut <= 0:
                cut = window.rfind("\n")
            if cut <= 0:
                cut = limit
            chunk, remaining = remaining[:cut], remaining[cut:].lstrip("\n")
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": chunk}})
    return blocks


def post_to_slack(channel: str, text: str, awk_id: str | None = None) -> str:
    """Post markdown to a Slack channel — or just log it when there's no
    SLACK_BOT_TOKEN (the demo's default). Returns the message ts (or "log").

    This is the ONLY path to Slack: the session object formats each kind of
    update (plan / status / report) into markdown and calls this. Pass `awk_id`
    for a plan that needs approval — in Slack that adds an Approve button plus an
    "Or chat about this:" hint (resolved by deploy/modal_app.py's
    /slack/interactivity), and in log mode it prints the resolve curl instead."""
    client = _slack_client()
    if client is None:
        if awk_id:
            text = f"{text}\n\n{_approval_curl(awk_id)}"
        logger.info("\n=== Slack (channel=%s) ===\n%s\n", channel, text)
        return "log"
    try:
        blocks = _mrkdwn_sections(text)
        if awk_id:
            blocks.append(_approve_button(awk_id))
            blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "Or chat about this:"}]})
        return client.chat_postMessage(channel=channel, text=text[:150], blocks=blocks)["ts"]
    except SlackApiError as e:
        err = e.response.get("error", "unknown") if e.response else "unknown"
        logger.warning("Slack post failed (%s) — logging instead.\n%s", err, text)
        return "log"


def _approve_button(awk_id: str) -> dict:
    """Single Approve action. The action_id carries the awakeable id; the click is
    resolved by deploy/modal_app.py's /slack/interactivity. There's no reject — to
    change the plan the user just sends a message, which steers the running run."""
    return {
        "type": "actions",
        "elements": [
            {"type": "button", "style": "primary", "value": "approve",
             "text": {"type": "plain_text", "text": "✅ Approve"},
             "action_id": f"plan_approve:{awk_id}"},
        ],
    }


def _approval_curl(awk_id: str) -> str:
    """Copy-pasteable curl to approve a plan parked on an awakeable — the log-mode
    fallback when there's no Slack button to click. To change the plan instead,
    send another message to the session and it steers the run."""
    url = f"{RESTATE_HOST}/restate/awakeables/{awk_id}/resolve"
    auth = "" if "localhost" in RESTATE_HOST else '-H "Authorization: Bearer $RESTATE_AUTH_TOKEN"'
    return (
        f"▶ Approve:  curl {url} {auth} --json '{{\"approved\": true}}'\n"
        f"▶ Or steer: send another message to the session to change the plan."
    )


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


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


async def llm_call(req: LLMRequest) -> dict:
    """Call the model and return just the assistant message dict
    (`{"role", "content", "tool_calls"?}`) — clean to append back to history."""
    if OFFLINE:
        response = await stub_provider(req)
    else:
        messages = ([{"role": "system", "content": req.prompt}] if req.prompt else []) + req.msgs
        completion = await acompletion(
            model=req.model,
            messages=messages,
            tools=req.tools,
            response_format=req.output_schema,
        )
        response = completion.model_dump(exclude_none=True)  # exclude_none -> clean message

    result =  response["choices"][0]["message"]
    if req.output_schema and not req.tools:
        return json.loads(result["content"])
    return result


def to_tool(fn) -> dict:
    """Wrap a plain async function as an OpenAI-style tool definition, deriving
    name/description/parameters from its signature and docstring."""
    return {"type": "function", "function": function_to_dict(fn)}


async def peek(fut: RestateDurableFuture[T]) -> T | None:
    assert isinstance(fut, ServerDurableFuture)
    if fut.is_completed():
        return await fut
    return None


def append(history: ChatHistory, plan: dict, sub_reports: list[dict], text: str) -> None:
    # subtopic -> its findings, so the planner sees what's done and only adds new subtopics.
    done = {sr["subtopic"]: sr["findings"] for sr in sub_reports}
    history.messages.append(assistant(
        f"Research already completed for '{plan['topic']}' (do not redo these subtopics):\n{json.dumps(done, indent=2)}"
    ))
    history.messages.append(user(
        f"Steering update from the user — incorporate this; only plan NEW research it requires:\n{text}"
    ))


def plan_request(history: ChatHistory):
    return LLMRequest(prompt=PLANNER, msgs=history.messages, output_schema=Plan)

def write_request(history: ChatHistory, plan: dict, sub_reports: list[dict]):
    return LLMRequest(prompt=WRITER, msgs=history.messages + [user(to_brief(plan, sub_reports))], output_schema=Report)

def classify_request(history: ChatHistory, text: str):
    return LLMRequest(model=FAST_MODEL, prompt=CLASSIFIER, output_schema=Strategy,
      msgs=[user(f"Current goal: {history.messages[-3:]} - New message: {text}")])