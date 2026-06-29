# A steerable, governed research agent with Restate + Tavily

A long-running deep-research agent (planner → parallel researchers → writer) that you can
**talk to while it works** and that routes **every model call through one governed gateway** —
built on [Restate](https://restate.dev). The demo is designed to show the two things that make
Restate different from "just a durable workflow engine":

1. **Steer / interrupt / enqueue a run that is already in flight.** Send a follow-up mid-run and
   it either folds into the live run (*steer*), cancels-and-restarts on a new goal (*interrupt*),
   or runs after (*enqueue*). Interrupt cancels the run **and its whole fan-out of researchers**
   in one durable signal — distributed stack-unwinding, with cleanup that always runs.
2. **An LLM gateway: model-policy + org-wide flow control.** Every model call goes through one
   `LLMGateway` service that enforces a model allow-list and runs inside a concurrency-limited
   *scope*, so one config line caps concurrent model spend across the whole org — no Redis, no
   semaphore.

The agent loop is hand-written against `litellm.acompletion`; durability, steering, state, and
flow control all come from Restate.

> The canonical demo lives in **`app_litellm/deep_research.py`**. The same workflow built with
> LangChain's `create_agent` is in `app/deep_research.py` (kept as a framework-interop reference).

## Architecture

Four Restate objects, keyed by the Slack thread / session id:

| Object | Kind | Role |
|--------|------|------|
| `Controller` | Virtual Object | Entry point. Owns the session history + the in-flight run's invocation id. Routes each new message to steer / interrupt / enqueue. |
| `DeepResearchAgent.research` | Virtual Object | The long-running run the Controller dispatches and tracks. Planner → approval → fan-out → writer. Cancellable. |
| `ResearchAgent.investigate` | Service | One subtopic researcher (its own durable agent loop). The workflow fans out N of these in parallel. |
| `LLMGateway.complete` | Service | Every model call goes through here: model allow-list policy, then the journaled provider call. Invoked through `ctx.scope("llm")` for flow control. |

```
Slack msg ─▶ Controller (history + steer/interrupt/enqueue)
                 │ object_send + track invocation id
                 ▼
            DeepResearchAgent.research ──▶ planner ─▶ [human approval]
                 │                                        │ fan out
                 │                                        ▼
                 │                         N × ResearchAgent.investigate
                 │                                        │
                 └────────────── every model call ───────┴──▶ ctx.scope("llm").service_call(LLMGateway.complete)
                                                                   │
                                                          model allow-list + concurrency cap
```

## The two differentiators in code

### 1. Steer / interrupt / enqueue (`Controller.message`)
A Virtual Object can't act on *itself* while busy (a second handler just queues). So the
Controller tracks the run as a *separate* invocation it can cancel or signal:

```python
current = await ctx.get("current", type_hint=str)
if current is None:
    await _start(ctx, history)                       # nothing running → just start
else:
    match await _classify(ctx, goal, text):
        case "interrupt":
            ctx.cancel_invocation(current)           # cancels the run AND its fan-out
            await _start(ctx, history)               # roll forward to the new goal
        case "steer":
            ctx.resolve_signal(current, "steer", text)   # fold into the live run
        case _:  # enqueue
            await _start(ctx, history)               # run VO serializes → runs after
```

The run catches the cancellation, runs durable cleanup, and re-raises:

```python
try:
    response = await deep_research(ctx, query, history)
except restate.TerminalError as e:
    if e.status_code == 409:                         # cancelled by an interrupt
        await ctx.run_typed("stopped", post_to_channel, channel=ctx.key(), text="⏹️ Stopped…")
    raise
```

### 2. LLM gateway = policy + flow control (`LLMGateway.complete`)
Every model call is routed through one scoped service call:

```python
raw = await ctx.scope("llm").service_call(complete, arg=LLMRequest(...), limit_key=...)
```

```python
@llm_gateway.handler()
async def complete(ctx, req: LLMRequest) -> dict:
    if req.model not in APPROVED_MODELS:                       # policy — instant, no spend
        raise restate.TerminalError(f"Model '{req.model}' is not approved")
    return await ctx.run_typed("provider", _provider_call, req=req)   # journaled call
```

A `ctx.run` step can't be flow-controlled; a scoped `service_call` can. With one rule —
`restate rules set llm --concurrency 3` — Restate caps concurrent model calls across every
session that shares the `"llm"` scope. Set `PER_SESSION_FAIRNESS = True` to instead give each
session its own queue (`limit_key=<session>`) so one busy session can't starve the others.

## Run locally

### 1. Setup
```bash
uv sync
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
# optional — wire to a real Slack channel; otherwise delivery prints to stdout
export SLACK_BOT_TOKEN=xoxb-...
```

> No internet or API keys? See **[Run offline](#run-offline-no-network--no-api-keys)** below —
> the whole demo runs with stubbed LLM + web-search calls.

### 2. Start the Restate Server (flow control enabled)
Flow control (scoped concurrency limits) is opt-in and only enables on a **fresh** server. It
also needs a **recent** server build — the `scope` feature requires invocation protocol **v7**.
An older server fails the model calls with *"Feature 'scope' is not supported by the negotiated
protocol version …v6…"*; pull the latest image to be safe:

```bash
docker pull docker.restate.dev/restatedev/restate:latest
docker run -p 8080:8080 -p 9070:9070 -p 9071:9071 \
  --add-host=host.docker.internal:host-gateway \
  -e RESTATE_EXPERIMENTAL_ENABLE_VQUEUES=true \
  docker.restate.dev/restatedev/restate:latest
```

> On an older server you can't upgrade? Run the app with `FLOW_CONTROL=0` — model calls then
> skip the scope, so everything works except the concurrency-cap beat (the gateway's policy and
> the governed service hop still run).

### 3. Run the app and register it
```bash
uv run app_litellm
```
In the UI (`http://localhost:9070`) register the deployment at `http://host.docker.internal:9080`.
You should see `Controller`, `DeepResearchAgent`, `ResearchAgent`, and `LLMGateway`.

### 4. Drive it
```bash
# Start a research run for session "demo"
curl localhost:8080/Controller/demo/message --json '"What is new in AI agents?"'

# Approve the plan when it posts (copy the awakeable curl from the service log):
curl localhost:8080/restate/awakeables/<awk_id>/resolve --json '{"approved": true}'

# While it runs, send a follow-up — the classifier picks steer / interrupt / enqueue:
curl localhost:8080/Controller/demo/message --json '"actually, focus only on coding agents"'   # steer
curl localhost:8080/Controller/demo/message --json '"forget that — research AI policy instead"' # interrupt
```

### 5. Demo the flow control
```bash
# Cap concurrent model calls across the org, then fire a swarm of runs
restate rules set llm --concurrency 3
scripts/swarm.sh 12

# Watch only 3 LLMGateway/complete invocations run at a time (rest queued) in the UI,
# or query the system tables:
#   SELECT * FROM sys_user_limits;   SELECT * FROM sys_vqueues;
```

### Other knobs
- **Resilience:** set `FAILURE_PROBABILITY = 0.2` in `app_litellm/utils/tools.py` to make Tavily
  flaky — the UI shows a tool step failing and retrying to a journaled success.
- **Policy:** point a request at a model outside `APPROVED_MODELS` to see the gateway refuse it
  without a provider call.
- **Autonomous mode:** `curl localhost:8080/DeepResearchAgent/demo/scan_news --json '"AI"'`
  posts a daily news digest and self-schedules tomorrow's run (no cron).

## Run offline (no network / no API keys)

For rehearsing the demo with no internet (e.g. on a plane), `OFFLINE=1` stubs every LLM and
web-search call with canned, **schema-valid** responses. The full demo runs unchanged — fan-out,
steer / interrupt / enqueue, flow control, retries, sessions — because the only thing that goes
out to the network (the model + Tavily) is replaced; all the Restate behavior is local anyway.

What the stub does:
- **Drives the loop like a real model would** — researchers emit a `web_search` tool call on the
  first turn, then a final structured answer, so the fan-out and durable tool steps still appear
  in the UI.
- **Deterministic steering** — the classifier picks the path from keywords in your message:
  *"also / focus / add / include / emphasize"* → **steer**, *"forget / instead / stop / different"*
  → **interrupt**, anything else → **enqueue**. So each beat is reproducible on stage.
- **Stays visible** — a per-call delay (`STUB_DELAY`, default `2` seconds) keeps the parallel
  researchers, the flow-control queue, and the interrupt window observable. Lower it for a faster
  rehearsal, raise it for a longer interrupt window.
- **Retries still work** — `FAILURE_PROBABILITY` in `app_litellm/utils/tools.py` makes the stubbed
  search flaky too, so the retry-to-journaled-success visual works offline.

**Prerequisite:** the Restate server runs in Docker, so pull the image *before* you go offline:

```bash
docker pull docker.restate.dev/restatedev/restate:latest
```

Then, fully offline:

```bash
# 1. Restate server (flow control enabled) — no internet needed once the image is pulled
docker run -p 8080:8080 -p 9070:9070 -p 9071:9071 \
  --add-host=host.docker.internal:host-gateway \
  -e RESTATE_EXPERIMENTAL_ENABLE_VQUEUES=true \
  docker.restate.dev/restatedev/restate:latest
  
restate rules set "llm-gateway" --concurrency 100 
restate rules set "agent/*" --concurrency 3 

# 2. The app in stub mode — no OPENAI_API_KEY / TAVILY_API_KEY required
OFFLINE=1 uv run app_litellm
# tune the pacing if you like: OFFLINE=1 STUB_DELAY=4 uv run app_litellm
# older cached server image (no protocol v7)? skip the scope: FLOW_CONTROL=0 OFFLINE=1 uv run app_litellm
```

Register `http://host.docker.internal:9080` in the UI (`http://localhost:9070`), then drive it
with the **exact same** commands as steps 4–5 above (run, steer/interrupt, `restate rules set llm
--concurrency 3`, `scripts/swarm.sh`). Everything works identically — only the answers are canned.

## Deploy to production
Ship on Restate Cloud (or self-hosted) + serverless functions (Modal, Render, Railway, …), with
Slack as the delivery channel.
