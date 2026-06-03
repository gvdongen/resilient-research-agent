# Deploy as a Slack bot on Restate Cloud + Modal

This folder turns the local agent (see the [project README](../README.md))
into a Slack bot driven by Restate Cloud + Modal. Nothing
to manage: Restate Cloud hosts the durable broker (journals, KV, timers,
awakeables, retries), Modal hosts the Python services as serverless
functions, and a thin Slack webhook in [`modal_app.py`](modal_app.py)
bridges Slack events to the Restate ingress.

> The agent itself does **not** depend on Slack — its delivery helpers
> fall back to log output if `SLACK_BOT_TOKEN` is unset. You can deploy
> the Restate service half (step 4 below) and skip the Slack app
> entirely; everything is still drivable via `curl` against the Restate
> Cloud ingress.

## 0. Sign up for Restate Cloud and Modal

Create a free account at [restate.dev](https://restate.dev) and
[Modal](https://modal.com).

## 1. Install the Modal CLI and log in

```bash
uv tool install modal
modal token new
```

## 2. Optional: Create the Slack app

Skip this step if you only want to drive the agent via curl — the
`/slack/*` endpoints will simply go unused. Otherwise, create a Slack
app so you can get the signing secret and bot token needed by the Modal
secret in the next step.

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New
   App** → **From scratch**. Pick a name and workspace.
2. Under **OAuth & Permissions** → **Bot Token Scopes**, add:
   - `chat:write` — post research plans, news cards, and final reports
   - `commands` — register the `/daily-report` slash command
   - `app_mentions:read` and `channels:history` (or `groups:history` for
     private channels) — receive channel messages that kick off research
3. Under **Basic Information**, copy the **Signing Secret** — you'll
   need it for `SLACK_SIGNING_SECRET` in the next step.
4. Under **Install App**, install to your workspace and copy the **Bot
   User OAuth Token** (starts with `xoxb-`) — that's `SLACK_BOT_TOKEN`.

Leave the Event Subscriptions, Interactivity, and Slash Commands pages
open in a tab — you'll fill in URLs in step 5 after Modal prints the
webhook URL.

## 3. Create a Modal secret with the API keys

```bash
modal secret create research-agent-secrets \
  OPENAI_API_KEY=sk-... \
  TAVILY_API_KEY=tvly-... \
  SLACK_BOT_TOKEN=xoxb-...                        # optional
  SLACK_SIGNING_SECRET=...                        # required for /slack/* endpoints
  RESTATE_CLOUD_INGRESS=https://<env>.env.<region>.restate.cloud:8080 \
  RESTATE_CLOUD_PUBLICKEY=publickeyv1_...         # to verify if requests are from your cloud account
  RESTATE_AUTH_TOKEN=<your-restate-api-key>
```

## 4. Deploy to Modal

From the **project root** (so `pyproject.toml` and `app/` resolve correctly):

```bash
modal deploy deploy/modal_app.py
```

Modal prints two URLs — one for the Restate services, one for the Slack
webhook. Save the Slack webhook URL (looks like
`https://<workspace>--research-agent-slack-webhook.modal.run`); you'll
paste it into the Slack app config next.

## 5. Point the Slack app at the Modal webhook

Skip if you didn't create a Slack app in step 2. Otherwise, head back to
your app at [api.slack.com/apps](https://api.slack.com/apps) and wire up
the three endpoints — all of them live under the Slack webhook URL Modal
printed in step 4.

- **Event Subscriptions** → toggle on → **Request URL**:
  `<slack-webhook-url>/slack/events`. Slack will hit it once with a
  `url_verification` challenge; the endpoint responds automatically.
  Under **Subscribe to bot events**, add `message.channels` (and
  `message.groups` for private channels). A channel message becomes a
  research run keyed on that channel.
- **Interactivity & Shortcuts** → toggle on → **Request URL**:
  `<slack-webhook-url>/slack/interactivity`. This is where Approve /
  Reject button clicks on plan cards resolve the awakeable.
- **Slash Commands** → **Create New Command**:
  - Command: `/daily-report`
  - Request URL: `<slack-webhook-url>/slack/commands`
  - Short description: `Start a daily news + research loop on a topic`
  - Usage hint: `<topic>`

After saving, **reinstall** the app to the workspace if Slack prompts
for it — adding scopes or a slash command invalidates the existing
install.

Invite the bot to whichever channel you want to drive (`/invite
@your-app`). Mention it, send a message, or run `/daily-report AI` to
kick off a run.

## 6. Register the deployment with Restate Cloud

Get an API key and ingress URL from the [Restate Cloud UI](https://cloud.restate.dev),
then register the services URL via the UI (Overview → Register Deployment).

All services show up in the Restate Cloud UI.

## 7. Invoke against Restate Cloud

Go to the Restate Cloud UI, click on the handler you want to invoke (e.g.
`DeepResearchAgent/<channel>/scan_news`), and send a message like `"AI"`.

Same payloads as locally, just swap the ingress and add the auth header:

```bash
export RESTATE_INGRESS=https://<env>.env.<region>.restate.cloud:8080
export RESTATE_AUTH_TOKEN=<your-restate-api-key>

curl $RESTATE_INGRESS/DeepResearchAgent/C123/scan_news \
  -H "Authorization: Bearer $RESTATE_AUTH_TOKEN" \
  --json '"AI"'
```
