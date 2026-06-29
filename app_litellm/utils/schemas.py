"""Shared pydantic models for the litellm flavor.

Same shapes as `app/utils/schemas.py`, except `ChatHistory.messages` holds
raw litellm-format dicts (`{"role": ..., "content": ..., ...}`) instead of
LangChain `AnyMessage` instances.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---- Chat ------------------------------------------------------------------


class ChatHistory(BaseModel):
    """Persistent multi-turn chat history for one DeepResearchAgent key."""

    messages: list[dict] = Field(default_factory=list)


# ---- LLM gateway -----------------------------------------------------------


class LLMRequest(BaseModel):
    """One model call routed through the LLMGateway service so it can be
    policy-checked (model allow-list) and flow-controlled (scope concurrency).
    Everything is plain JSON so it crosses the service-call boundary cleanly."""

    model: str
    messages: list[dict]
    tools: list[dict] | None = None
    output: dict | None = None


# ---- Controller ------------------------------------------------------------


class StrategyChoice(BaseModel):
    """How to handle a message that arrives while a run is already in flight."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["enqueue", "interrupt", "steer"]
    reason: str


# ---- Planner ---------------------------------------------------------------


class Plan(BaseModel):
    """Output of the planner: angle + subtopics for parallel research."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    rationale: str
    subtopics: list[str]


class Decision(BaseModel):
    """Human verdict on a proposed ResearchPlan, delivered via a Slack button.

    Reject carries no notes: the human just types their feedback as the next
    channel message, which re-runs the handler with the rejected plan in view."""

    model_config = ConfigDict(extra="forbid")

    approved: bool


# ---- Researchers -----------------------------------------------------------


class SubReport(BaseModel):
    """One researcher's findings for a single subtopic."""

    model_config = ConfigDict(extra="forbid")

    subtopic: str
    findings: str
    sources: list[str]


# ---- Writer ----------------------------------------------------------------


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str
    body: str


class Report(BaseModel):
    """Output of the ReportWriter."""

    model_config = ConfigDict(extra="forbid")

    headline: str
    executive_summary: str
    sections: list[Section]
    sources: list[str]


# ---- News scout ------------------------------------------------------------


class NewsItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str
    summary: str
    url: str


class NewsDigest(BaseModel):
    """Output of the news scout: today's news on the topic."""

    model_config = ConfigDict(extra="forbid")

    overview: str
    items: list[NewsItem]
