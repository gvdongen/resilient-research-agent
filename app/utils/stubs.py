"""Offline-demo stubs for the LLM gateway.

When `OFFLINE=1`, `deep_research._provider_call` delegates here instead of calling the
model, so the whole demo runs with no network or API keys. Responses are canned but
realistic and themed around "what's new in AI agents", and the stub drives the agent
loop the way a real model would: emit a `web_search` tool call on the first
researcher/news turn, then a final structured answer.

A small per-call delay (`STUB_DELAY` seconds) keeps the parallel researchers, the
flow-control queue, and the steer/cancel window visible in the UI.
"""

import asyncio
import json
import os
import random

from .schemas import (
    Report,
    LLMRequest,
    NewsDigest,
    NewsItem,
    SubReport,
    Plan,
    Section,
    Strategy,
)

STUB_DELAY = float(os.environ.get("STUB_DELAY", "2"))

# Canned-but-real content so the stubbed run reads like a genuine research result rather than
# placeholder text. Each entry is (findings, sources). Plan subtopics are the keys, so each
# researcher's report matches its assigned topic.
_AI_FINDINGS: dict[str, tuple[str, list[str]]] = {
    "Agentic coding tools": (
        "Coding agents moved from autocomplete to autonomous task execution. Tools like "
        "Claude Code, Cursor, and OpenAI's Codex now plan multi-step changes, run tests, and "
        "open pull requests with light supervision, and increasingly work asynchronously in the "
        "background. Frontier models pushed SWE-bench Verified well past 70%, shifting the focus "
        "from single edits to long-horizon engineering tasks and review-in-the-loop workflows.",
        [
            "https://www.anthropic.com/claude-code",
            "https://openai.com/index/introducing-codex/",
            "https://www.swebench.com",
        ],
    ),
    "Tool standardization with MCP": (
        "The Model Context Protocol (MCP), introduced by Anthropic in late 2024, has become the "
        "de facto open standard for connecting agents to tools and data. A large ecosystem of MCP "
        "servers now exposes databases, SaaS APIs, and filesystems to any MCP-capable client, and "
        "the major SDKs (Anthropic, OpenAI, Google) ship first-class MCP support — so a tool built "
        "once works across agents and vendors.",
        [
            "https://modelcontextprotocol.io",
            "https://www.anthropic.com/news/model-context-protocol",
        ],
    ),
    "Long-running & multi-agent orchestration": (
        "Production agents increasingly run for minutes to days, coordinating sub-agents and "
        "external systems. Orchestration moved up the stack with LangGraph, the OpenAI Agents SDK, "
        "and Google's ADK, while durable-execution runtimes (Restate, Temporal) handle retries, "
        "state, and recovery across long runs. The consensus this year: reliability and "
        "orchestration — not raw model capability — are the main blockers to shipping agents.",
        [
            "https://www.restate.dev",
            "https://langchain-ai.github.io/langgraph/",
            "https://platform.openai.com/docs/guides/agents",
        ],
    ),
}


def last_topic(messages: list[dict]) -> str:
    for m in reversed(messages):
        c = m.get("content") or ""
        if m.get("role") == "user" and c.startswith("Topic:"):
            return c[len("Topic:"):].strip()
    return "the requested topic"


def stub_strategy(messages: list[dict]) -> str:
    """Pick cancel/steer from keywords so both paths demo deterministically."""
    text = (messages[-1].get("content") or "").lower()
    if any(w in text for w in ("forget", "instead", "stop", "never mind", "cancel", "different")):
        return "cancel"
    return "steer"


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return "the requested topic"


def _called_tools(messages: list[dict]) -> set[str]:
    """Names of every tool the assistant has already called in this thread."""
    return {tc["function"]["name"] for m in messages for tc in (m.get("tool_calls") or [])}


def _tool_call(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": f"call_{name}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }


def stub_orchestrator(messages: list[dict]) -> dict:
    """Drive the orchestrator agent loop deterministically: plan, then research,
    then write — one tool call per turn, in order. A steer folded into the
    conversation (a "New input from the user" message) triggers one fresh
    create_plan, so re-planning on steer is visible offline."""
    calls = [tc["function"]["name"] for m in messages for tc in (m.get("tool_calls") or [])]
    steers = sum(1 for m in messages
                 if m.get("role") == "user" and "New input from the user" in (m.get("content") or ""))

    # (Re)plan whenever an un-absorbed steer is outstanding — including the first
    # turn, when there are no plans and no steers yet (0 >= 0).
    if steers >= calls.count("create_plan"):
        return _tool_call("create_plan", {"topic": _last_user(messages)})

    since_plan = calls[len(calls) - calls[::-1].index("create_plan"):]
    if "run_research" not in since_plan:
        return _tool_call("run_research", {"subtopics": list(_AI_FINDINGS)})
    if "write_report" not in since_plan:
        return _tool_call("write_report", {})
    return {"role": "assistant", "content": "Research complete."}


def stub_content(name: str, messages: list[dict]) -> str:
    """Canned, schema-valid JSON for the requested response model."""
    topic = last_topic(messages)
    if name == "Plan":
        return Plan(
            topic="What's new in AI agents",
            rationale=(
                "AI agents crossed from demos into production over the past year. The most "
                "consequential threads are how agents write software, how they connect to tools, "
                "and how they run reliably at scale."
            ),
            subtopics=list(_AI_FINDINGS),
        ).model_dump_json()
    if name == "SubReport":
        findings, sources = _AI_FINDINGS.get(
            topic,
            (
                "AI agents are moving from chat assistants to autonomous systems that use tools, "
                "run for long periods, and coordinate with other agents and humans.",
                ["https://www.anthropic.com/news", "https://openai.com/news/"],
            ),
        )
        return SubReport(subtopic=topic, findings=findings, sources=sources).model_dump_json()
    if name == "Report":
        return Report(
            headline="What's new in AI agents (mid-2026)",
            executive_summary=(
                "AI agents crossed from demos into production over the past year. Three shifts "
                "stand out: coding agents that complete real engineering tasks with light "
                "supervision, MCP emerging as the universal way to connect agents to tools and "
                "data, and a wave of orchestration and durable-execution infrastructure for "
                "running agents reliably at scale."
            ),
            sections=[
                Section(heading=t, body=findings)
                for t, (findings, _) in _AI_FINDINGS.items()
            ],
            sources=[s for _, srcs in _AI_FINDINGS.values() for s in srcs][:5],
        ).model_dump_json()
    if name == "NewsDigest":
        return NewsDigest(
            overview=(
                "Recent moves in AI agents center on autonomous coding, the fast-growing MCP tool "
                "ecosystem, and infrastructure for running agents reliably."
            ),
            items=[
                NewsItem(
                    headline="Coding agents open PRs end-to-end",
                    summary="Claude Code, Cursor, and Codex now plan changes, run tests, and submit pull requests with light supervision.",
                    url="https://www.anthropic.com/claude-code",
                ),
                NewsItem(
                    headline="MCP becomes the standard for agent tools",
                    summary="The Model Context Protocol ecosystem keeps expanding, with first-class support across the major agent SDKs.",
                    url="https://modelcontextprotocol.io",
                ),
                NewsItem(
                    headline="Durable execution gains ground for agents",
                    summary="Teams adopt runtimes like Restate to make long-running, stateful agents resilient to failures and restarts.",
                    url="https://www.restate.dev",
                ),
            ],
        ).model_dump_json()
    if name == "Strategy":
        return Strategy(strategy=stub_strategy(messages)).model_dump_json()
    return "{}"


async def stub_provider(req: LLMRequest) -> dict:
    """Drop-in replacement for the provider call: returns a ModelResponse-shaped dict."""
    # Delay so parallel researchers, the concurrency cap, and the steer/cancel window are visible.
    await asyncio.sleep(random.uniform(STUB_DELAY * 0.5, STUB_DELAY * 15))
    tool_names = {t["function"]["name"] for t in (req.tools or [])}
    if "create_plan" in tool_names:  # the orchestrator agent loop
        return {"choices": [{"message": stub_orchestrator(req.msgs)}]}

    name = (req.output_schema or {}).get("json_schema", {}).get("name", "")
    already_searched = any(m.get("role") == "tool" for m in req.msgs)
    if req.tools and not already_searched:
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_stub",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"queries": [last_topic(req.msgs)], "time_range": "month"}),
                    },
                }
            ],
        }
    else:
        message = {"role": "assistant", "content": stub_content(name, req.msgs)}
    return {"choices": [{"message": message}]}
