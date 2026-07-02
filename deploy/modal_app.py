"""Deploy the Restate services from `app/` to Modal as one ASGI endpoint.

Register the resulting URL with Restate Cloud (or any Restate server) as a
deployment. Restate then calls back into this endpoint over HTTP/2 to drive
every handler.

Run from the project root so `pyproject.toml` and `app/` resolve:

    modal deploy deploy/modal_app.py
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
    import os
    import sys
    sys.path.insert(0, "/root/app")

    import restate
    from session_coordinator import controller
    from deep_research import deep_research_agent, deep_research_agent_v2, research_agent
    from llm_gateway import llm_gateway

    return restate.app(
        # Modal serves ASGI request/response, not a true bidirectional HTTP/2 stream.
        # With protocol="bidi" Restate waits out its inactivity timeout (~1 min) at
        # every suspension point because Modal buffers the streamed response — the
        # source of the ~60s stall before each LLM call. Request/Response mode (as
        # used for Lambda) invokes the handler per step and ignores that timeout.
        services=[controller, deep_research_agent, deep_research_agent_v2, research_agent, llm_gateway],
        protocol="bidi",
        identity_keys=[os.environ["RESTATE_CLOUD_PUBLICKEY"]],
    )


@app.function(
    secrets=[modal.Secret.from_name("research-agent-secrets")],
    min_containers=1,
)
@modal.asgi_app()
def slack_webhook():
    """Slack-facing endpoints:

    - /slack/events        → channel message → Controller/{channel}/message (the session coordinator)
    - /slack/commands      → /daily-report — not available in the app build (no scan_news)
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

        # Every channel message goes to the Controller (the session coordinator),
        # keyed by channel. It owns the session history + the in-flight run and
        # decides whether to start a run, steer it, or cancel it.
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{restate_ingress}/Controller/{channel}/message/send",
                headers={"Authorization": f"Bearer {restate_auth}"},
                json=text,
            )
        return {}

    @api.post("/slack/commands")
    async def commands(req: Request):
        """`/daily-report` — the autonomous news-digest mode (`scan_news`) doesn't
        exist in the app build, so this command isn't wired to a run. We
        still verify the signature and reply gracefully (rather than 404) so the
        Slack app config can stay unchanged."""
        ts = req.headers.get("x-slack-request-timestamp", "")
        body = await req.body()
        if not verify(body, ts, req.headers.get("x-slack-signature", "")):
            raise HTTPException(401, "bad signature")

        return {
            "response_type": "ephemeral",
            "text": "`/daily-report` isn't available in this deployment — just send a message "
                    "in the channel to start a research run instead.",
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
        """Approve button clicks from plan-approval cards."""
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

        return {}

    return api