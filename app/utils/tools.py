import os
from slack_sdk import WebClient
from .schemas import NewsDigest, ResearchPlan, SubReport, FinalReport

slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

def summarize(topic: str, news: NewsDigest) -> str:
    news_lines = "\n".join(
        f"- {it.headline}: {it.summary} ({it.url})"
        for it in news.items
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
    resolve_url = f"http://localhost:8080/restate/awakeables/{awk_id}/resolve"
    yes_cmd = f"curl {resolve_url} --json '\"Tell me more about ...\"'"

    items_md = "\n\n".join(
        f"*{i.headline}*\n{i.summary}\n<{i.url}>"
        for i in digest.items
    )

    blocks: list[dict] = [
        {"type": "header",
         "text": {"type": "plain_text", "text": f"Today's news: {topic}"}},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": f"_{digest.overview}_"}},
        {"type": "divider"},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": items_md}},
        {"type": "divider"},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": (
             "▶ *Want to dive deeper?*\n\n"
             f"```{yes_cmd}```\n\n"
         )}},
    ]
    resp = slack_client.chat_postMessage(
        channel=os.environ["SLACK_DIGEST_CHANNEL_ID"],
        text=f"Today's news: {topic} — dive deeper?",
        blocks=blocks,
    )
    return resp["ts"]


def post_report(topic: str, report: FinalReport) -> str:
    """Post the FinalReport (rich Block Kit). Returns the message ts."""
    blocks: list[dict] = [
        {"type": "header",
         "text": {"type": "plain_text", "text": f"Deep research: {topic}"}},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": f"*{report.headline}*"}},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": report.executive_summary}},
        {"type": "divider"},
    ]
    for sec in report.sections:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f"*{sec.heading}*\n{sec.body}"},
        })
    if report.sources:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn",
                          "text": "Sources:\n" + "\n".join(f"• <{s}>" for s in report.sources)}],
        })

    resp = slack_client.chat_postMessage(
        channel=os.environ["SLACK_DIGEST_CHANNEL_ID"],
        text=report.headline,
        blocks=blocks,
    )
    return resp["ts"]
