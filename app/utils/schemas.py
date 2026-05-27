"""Shared pydantic models for all three phases."""
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field

# ---- Chat ----------------------------------------------------------------


class ChatHistory(BaseModel):
    """Persistent multi-turn chat history for one DeepResearchAgent key."""

    messages: list[AnyMessage] = Field(default_factory=list)


# ---- Planner ----------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Output of PlannerAgent: angle + subtopics for parallel research."""

    model_config = ConfigDict(extra="forbid")

    rationale: str
    subtopics: list[str]


class PlanDecision(BaseModel):
    """Human verdict on a proposed ResearchPlan, delivered via a Slack button.

    Reject carries no notes: the human just types their feedback as the next
    channel message, which re-runs the handler with the rejected plan in view."""

    model_config = ConfigDict(extra="forbid")

    approved: bool


# ---- Researchers ------------------------------------------------------------


class Report(BaseModel):
    """One researcher's findings for a single subtopic."""

    model_config = ConfigDict(extra="forbid")

    subtopic: str
    findings: str
    sources: list[str]


# ---- Writer -----------------------------------------------------------------


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str
    body: str


class FinalReport(BaseModel):
    """Output of the ReportWriter."""

    model_config = ConfigDict(extra="forbid")

    headline: str
    executive_summary: str
    sections: list[Section]
    sources: list[str]


# ---- News scout -------------------------------------------------------------


class NewsItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str
    summary: str
    url: str


class NewsDigest(BaseModel):
    """Output of NewsScoutAgent: today's news on the topic."""

    model_config = ConfigDict(extra="forbid")

    overview: str
    items: list[NewsItem]