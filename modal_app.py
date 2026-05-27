"""Deploy the Restate services from `app/` to Modal as one ASGI endpoint.

Register the resulting URL with Restate Cloud (or any Restate server) as a
deployment. Restate then calls back into this endpoint over HTTP/2 to drive
every handler.

    modal deploy modal_app.py
    restate dp register https://<org>--deep-research-agent-restate-services.modal.run
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.14")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("fastapi", "httpx")
    .add_local_dir("app", remote_path="/root/app")
)

app = modal.App("deep-research-agent", image=image)


@app.function(
    secrets=[modal.Secret.from_name("research-agent-secrets")],
    min_containers=1,
    timeout=36000,
)
@modal.asgi_app()
def restate_services():
    import sys
    sys.path.insert(0, "/root/app")

    import restate
    from deep_research import deep_research_agent, research_agent

    return restate.app(
        services=[deep_research_agent, research_agent],
        protocol="bidi",
	    identity_keys=["publickeyv1_8c7sHaJgEwqgn6PV1ywre8VkNVyM1HW3mBAJZ1WXwkSd"]
    )


@app.function(
    secrets=[modal.Secret.from_name("research-agent-secrets")],
    min_containers=1,
)
@modal.asgi_app()
def slack_webhook():
    """Slack-facing endpoints:

    - /slack/events        → channel message → DeepResearchAgent/{channel}/research
    - /slack/commands      → /daily-report <topic> → DeepResearchAgent/scan_news (idempotent per topic)
    - /slack/interactivity → plan_approve / plan_reject button clicks resolve
                             the plan-approval awakeable
    """
    import hashlib
    import hmac
    import json
    import os
    import time
    from urllib.parse import parse_qs

    import httpx
    from fastapi import FastAPI, HTTPException, Request

    signing_secret = os.environ["SLACK_SIGNING_SECRET"].encode()
    restate_ingress = os.environ["RESTATE_CLOUD_INGRESS"].rstrip("/")
    restate_auth = os.environ["RESTATE_AUTH_TOKEN"]
    slack_token = os.environ.get("SLACK_BOT_TOKEN")

    api = FastAPI()

    async def slack_post(client: httpx.AsyncClient, channel: str, text: str, thread_ts: str | None = None):
        """Best-effort chat.postMessage. Silently no-ops if SLACK_BOT_TOKEN unset."""
        if not slack_token:
            return
        body = {"channel": channel, "text": text}
        if thread_ts:
            body["thread_ts"] = thread_ts
        await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {slack_token}"},
            json=body,
        )

    def verify(body: bytes, ts: str, sig: str) -> bool:
        if not ts or not sig or abs(time.time() - int(ts)) > 60 * 5:
            return False
        base = b"v0:" + ts.encode() + b":" + body
        expected = "v0=" + hmac.new(signing_secret, base, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)

    @api.post("/slack/events")
    async def events(req: Request):
        body = await req.body()
        if not verify(
            body,
            req.headers.get("x-slack-request-timestamp", ""),
            req.headers.get("x-slack-signature", ""),
        ):
            raise HTTPException(401, "bad signature")

        payload = await req.json()
        if payload.get("type") == "url_verification":
            return {"challenge": payload["challenge"]}

        event = payload.get("event", {})
        if event.get("bot_id") or event.get("subtype"):
            return {}  # ignore bot messages, edits, joins — prevents loops

        channel = event.get("channel")
        text = event.get("text")
        if not channel or not text:
            return {}

        # ResearchSession was folded into the DeepResearchAgent VO (keyed by
        # channel), so a plain channel message kicks off a stateful research run.
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{restate_ingress}/DeepResearchAgent/{channel}/research/send",
                headers={"Authorization": f"Bearer {restate_auth}"},
                json=text,
            )
        return {}

    @api.post("/slack/commands")
    async def commands(req: Request):
        """`/daily-report <topic>` → kick off DeepResearchAgent/scan_news."""
        ts = req.headers.get("x-slack-request-timestamp", "")
        body = await req.body()
        if not verify(body, ts, req.headers.get("x-slack-signature", "")):
            raise HTTPException(401, "bad signature")

        form = parse_qs(body.decode())
        team_id = form.get("team_id", [""])[0]
        channel_id = form.get("channel_id", [""])[0]
        topic = form.get("text", [""])[0].strip()
        if not topic or not channel_id:
            return {"response_type": "ephemeral", "text": "Usage: `/daily-report <topic>`"}

        # Key on the Slack request `ts`, which stays constant across Slack's
        # automatic retries of one command click — so retries dedup, but a
        # fresh click (new ts) always kicks off a new run.
        idem_key = f"{team_id}:{channel_id}:{ts}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{restate_ingress}/DeepResearchAgent/{channel_id}/scan_news/send",
                headers={
                    "Authorization": f"Bearer {restate_auth}",
                    "idempotency-key": idem_key,
                },
                json=topic,
            )
        return {
            "response_type": "ephemeral",
            "text": f"📅 Daily report on *{topic}* scheduled. First card lands within a minute.",
        }

    async def resolve_awakeable(client: httpx.AsyncClient, awk_id: str, value) -> bool:
        """Resolve a Restate awakeable with `value`. Returns True on success."""
        resp = await client.post(
            f"{restate_ingress}/restate/awakeables/{awk_id}/resolve",
            headers={"Authorization": f"Bearer {restate_auth}"},
            json=value,
        )
        return resp.is_success

    @api.post("/slack/interactivity")
    async def interactivity(req: Request):
        """Button clicks from plan-approval and news cards."""
        body = await req.body()
        if not verify(
            body,
            req.headers.get("x-slack-request-timestamp", ""),
            req.headers.get("x-slack-signature", ""),
        ):
            raise HTTPException(401, "bad signature")

        # Slack sends `payload=<urlencoded json>`
        form = parse_qs(body.decode())
        raw = form.get("payload", [""])[0]
        if not raw:
            return {}
        payload = json.loads(raw)

        action = payload["actions"][0]
        action_id = action["action_id"]
        channel_id = payload["channel"]["id"]
        message_ts = payload["container"]["message_ts"]

        async with httpx.AsyncClient(timeout=5.0) as client:
            # ---- Approve the plan -------------------------------------------
            if action_id.startswith("plan_approve:"):
                awk_id = action_id.split(":", 1)[1]
                ok = await resolve_awakeable(client, awk_id, {"approved": True})
                await slack_post(
                    client,
                    channel_id,
                    "✅ Plan approved — researching now. The report will land here when done."
                    if ok
                    else "⏰ This plan request expired. Start a new run.",
                    thread_ts=message_ts,
                )
                return {}

            # ---- Reject the plan → ask for feedback in the channel ----------
            # The human just types what to change; that message re-runs the VO
            # handler with the rejected plan in view, so the planner revises.
            if action_id.startswith("plan_reject:"):
                awk_id = action_id.split(":", 1)[1]
                ok = await resolve_awakeable(client, awk_id, {"approved": False})
                await slack_post(
                    client,
                    channel_id,
                    "✏️ Plan rejected — reply here with what to change and I'll revise it."
                    if ok
                    else "⏰ This plan request expired. Start a new run.",
                    thread_ts=message_ts,
                )
                return {}

        return {}

    return api
