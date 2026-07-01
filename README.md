# A steerable, governed research agent with Restate + Tavily

A long-running deep-research agent (planner → parallel researchers → writer) that you can
**talk to while it works** and that routes **every model call through one governed gateway** —
built on [Restate](https://restate.dev). The demo is designed to show the two things that make
Restate different from "just a durable workflow engine":

1. **Steer or cancel a run that is already in flight.** Send a follow-up mid-run and it either
   folds into the live run (*steer*) or cancels-and-restarts on a new goal (*cancel*). Cancel
   tears down the run **and its whole fan-out of researchers** in one durable signal —
   distributed stack-unwinding, with cleanup that always runs.
2. **An LLM gateway: model-policy + org-wide flow control.** The workflow's model calls go through
   one `LLMGateway` service that enforces a model allow-list and runs inside a concurrency-limited
   *scope*, so one config line caps concurrent model spend across the whole department — no Redis,
   no semaphore.

The agent loop is hand-written against `litellm.acompletion`; durability, steering, state, and
flow control all come from Restate.

> The canonical demo lives in **`app/deep_research.py`**. The same workflow built with
> LangChain's `create_agent` is in `app/deep_research.py` (kept as a framework-interop reference).

## Architecture

A handful of Restate services and virtual objects, keyed by the Slack thread / session id:

| Object | Kind | Role |
|--------|------|------|
| `Controller` | Virtual Object | Entry point. Owns the session history + the in-flight run's invocation id. Routes each new message to steer or cancel. |
| `DeepResearchAgent.research` | Virtual Object | The long-running run the Controller dispatches and tracks. Planner → approval → fan-out → writer, wrapped in a steer/cancel loop. |
| `ResearchAgent.investigate` | Service | One subtopic researcher (its own durable agent loop). The workflow fans out N of these in parallel. |
| `LLMGateway.call_llm` | Service | The workflow's model calls go through here: model allow-list policy, then the journaled provider call. Invoked through `scope("department1")` for flow control. |

> A second virtual object, `DeepResearchAgentV1.research_v1`, is the same workflow at stage 1 —
> plan → approval → fan-out → write with the model calls made **locally** via `restate.run_typed`
> (no gateway, no steering). It's kept side-by-side with the full `DeepResearchAgent` so the demo
> can show the two stages.

```
Slack msg ─▶ Controller (history + steer/cancel)
                 │ object_send + track invocation id
                 ▼
            DeepResearchAgent.research ──▶ planner ─▶ [human approval]
                 │                                        │ fan out
                 │                                        ▼
                 │                         N × ResearchAgent.investigate
                 │                                        │
                 └─── plan / write / classify calls ──────┴──▶ scope("department1").service_call(LLMGateway.call_llm)
                                                                   │
                                                          model allow-list + concurrency cap
```

## The two differentiators in code

### 1. Steer or cancel (`Controller.message`)
A Virtual Object can't act on *itself* while busy (a second handler just queues). So the
Controller tracks the run as a *separate* invocation it can cancel or signal. If a message
arrives while a run is in flight, a fast classifier picks one of two strategies:

```python
current = await restate.get("current", type_hint=str)
if current is not None:                                       # a run is already in flight
    decision = await call_llm_gateway(restate, classify_request(history.messages, text))
    match decision["strategy"]:
        case "cancel":
            restate.cancel_invocation(current)                # cancels the run AND its fan-out
            # falls through → start a fresh run on the new goal
        case _:  # steer
            restate.resolve_signal(current, "steer", text)    # fold into the live run
            return
# nothing running (or we just cancelled) → start a run and remember its invocation id
handle = restate.object_send(research, key=restate.key(), arg=history)
restate.set("current", await handle.invocation_id())
```

`cancel` tears the running invocation — and its whole fan-out of researchers — down in one
durable signal, then rolls forward to the new goal. `steer` leaves the run alive and hands it
new context through a signal it picks up at its next checkpoint.

### 2. LLM gateway = policy + flow control (`LLMGateway.call_llm`)
The workflow never calls the model directly — it goes through the gateway over a scoped,
durable RPC (`llm_gateway.py`):

```python
async def call_llm_gateway(restate, req: LLMRequest) -> dict:
    return await restate.scope(DEPARTMENT).service_call(call_llm, arg=req)   # DEPARTMENT = "department1"

@llm_gateway.handler()
async def call_llm(restate, req: LLMRequest) -> dict:
    if req.model not in APPROVED_MODELS:                       # policy — instant, no spend
        raise rst.TerminalError(f"Model '{req.model}' is not on the approved list ...")
    return await restate.run_typed("provider", provider_call, req=req)   # journaled call
```

A plain `restate.run` step can't be flow-controlled; a scoped `service_call` can. With one rule —
`restate rules set department1 --concurrency 3` — Restate caps concurrent model calls across every
session that shares the `department1` scope, with no Redis and no semaphore. Pass a `limit_key` to
the `service_call` to give each key (e.g. each session or employee) its own fair queue, so one busy
user can't starve the others.

> The researcher sub-agents (`ResearchAgent.investigate`) currently journal their own model calls
> **locally** through the durable middleware rather than the gateway. Route them through the gateway
> too by passing `call_llm=call_llm` to `RestateMiddleware(...)`.

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

> The gateway always routes through `scope("department1")`, so a recent server (protocol v7) is
> required for the model calls to succeed — there's no toggle to skip the scope.

### 3. Run the app and register it
```bash
uv run app
```
In the UI (`http://localhost:9070`) register the deployment at `http://host.docker.internal:9080`.
You should see `Controller`, `DeepResearchAgent`, `DeepResearchAgentV1`, `ResearchAgent`, and `LLMGateway`.

### 4. Drive it
```bash
# Start a research run for session "demo"
curl localhost:8080/Controller/demo/message --json '"What is new in AI agents?"'

# Approve the plan when it posts (copy the awakeable curl from the service log):
curl localhost:8080/restate/awakeables/<awk_id>/resolve --json '{"approved": true}'

# While it runs, send a follow-up — the classifier picks steer or cancel:
curl localhost:8080/Controller/demo/message --json '"actually, focus only on coding agents"'   # steer
curl localhost:8080/Controller/demo/message --json '"forget that — research AI policy instead"' # cancel
```

### 5. Demo the flow control
```bash
# Cap concurrent model calls across the org, then fire a swarm of runs
restate rules set department1 --concurrency 3 
scripts/swarm.sh 12
```

### Other knobs
- **Resilience:** set `FAILURE_PROBABILITY = 0.2` in `app/utils/tools.py` to make Tavily
  flaky — the UI shows a tool step failing and retrying to a journaled success.
- **Policy:** point a request at a model outside `APPROVED_MODELS` to see the gateway refuse it
  without a provider call.
- **Two stages:** to demo stage 1, point the Controller at the simpler run — in
  `session_coordinator.py` swap `from deep_research import research` for
  `from deep_research import research_v1 as research`. Stage 1 is plan → approval → fan-out →
  write with local model calls (`restate.run_typed`), no gateway and no steering.

## Run offline (no network / no API keys)

For rehearsing the demo with no internet (e.g. on a plane), `OFFLINE=1` stubs every LLM and
web-search call with canned, **schema-valid** responses. The full demo runs unchanged — fan-out,
steer / cancel, flow control, retries, sessions — because the only thing that goes
out to the network (the model + Tavily) is replaced; all the Restate behavior is local anyway.

What the stub does:
- **Drives the loop like a real model would** — researchers emit a `web_search` tool call on the
  first turn, then a final structured answer, so the fan-out and durable tool steps still appear
  in the UI.
- **Deterministic steering** — the classifier picks the path from keywords in your message:
  *"forget / instead / stop / cancel / different"* → **cancel**, anything else → **steer**.
  So each beat is reproducible on stage.
- **Stays visible** — a per-call delay (`STUB_DELAY`, default `2` seconds) keeps the parallel
  researchers, the flow-control queue, and the steer/cancel window observable. Lower it for a
  faster rehearsal, raise it for a longer window.
- **Retries still work** — `FAILURE_PROBABILITY` in `app/utils/tools.py` makes the stubbed
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

# cap concurrent model calls for the department scope
restate rules set department1 --concurrency 3

# 2. The app in stub mode — no OPENAI_API_KEY / TAVILY_API_KEY required
OFFLINE=1 uv run app
# tune the pacing if you like: OFFLINE=1 STUB_DELAY=4 uv run app
```

Register `http://host.docker.internal:9080` in the UI (`http://localhost:9070`), then drive it
with the **exact same** commands as steps 4–5 above (run, steer/cancel, `restate rules set
department1 --concurrency 3`, `scripts/swarm.sh`). Everything works identically — only the answers are canned.

## Deploy to production
Ship on Restate Cloud (or self-hosted) + serverless functions (Modal, Render, Railway, …), with
Slack as the delivery channel.
