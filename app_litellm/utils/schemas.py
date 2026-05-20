"""Shared pydantic models for the deep-research pipeline."""

from pydantic import BaseModel, ConfigDict


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


# ---- Planner ----------------------------------------------------------------

class PlanRequest(BaseModel):
    """Input to PlannerAgent.plan — topic plus today's news for context."""
    model_config = ConfigDict(extra="forbid")

    topic: str
    news: NewsDigest


class ResearchPlan(BaseModel):
    """Output of PlannerAgent: angle + subtopics for parallel research."""
    model_config = ConfigDict(extra="forbid")

    rationale: str
    subtopics: list[str]


# ---- Researchers ------------------------------------------------------------

class SubReport(BaseModel):
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


class WriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str
    plan: ResearchPlan
    sub_reports: list[SubReport]


# ---- Orchestrator return ----------------------------------------------------

class DailyResult(BaseModel):
    """What the orchestrator returns each day: always the news, optionally a deep report."""
    model_config = ConfigDict(extra="forbid")

    news: NewsDigest
    report: FinalReport | None = None
