"""Shared pydantic models for the litellm flavor.

Same shapes as `app/utils/schemas.py`, except `ChatHistory.messages` holds
raw litellm-format dicts (`{"role": ..., "content": ..., ...}`) instead of
LangChain `AnyMessage` instances.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---- Chat ------------------------------------------------------------------


class ChatHistory(BaseModel):
    """Persistent multi-turn chat history for one DeepResearchAgent key."""

    messages: list[dict] = Field(default_factory=list)


# ---- LLM gateway -----------------------------------------------------------


class LLMRequest(BaseModel):
    model: str = "gpt-4o-mini"
    prompt: str | None = None
    msgs: list[dict]
    tools: list[dict] | None = None
    output_schema: dict | None = None

    @field_validator("output_schema", mode="before")
    @classmethod
    def _coerce_output(cls, v):
        """Accept a pydantic model class and turn it into OpenAI strict
        json_schema. Strict json_schema requires `extra="forbid"` and no
        defaults, which our output models declare. A dict passes through
        unchanged (e.g. after deserialization across the service boundary)."""
        if isinstance(v, type) and issubclass(v, BaseModel):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": v.__name__,
                    "schema": v.model_json_schema(),
                    "strict": True,
                },
            }
        return v


# ---- Controller ------------------------------------------------------------


class Strategy(BaseModel):
    """How to handle a message that arrives while a run is already in flight."""
    model_config = ConfigDict(extra="forbid")
    cancel: Literal["cancel", "steer"]


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


class Topic(BaseModel):
    session: str
    topic: str


class Brief(BaseModel):
    """A research brief handed to the writer sub-agent."""

    session: str
    brief: str


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
