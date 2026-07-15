# A steerable deep research agent on Restate

A long-running deep-research agent — **plan → human approval → parallel researchers → writer** —
modeled as a stateful, persistent entity you can interact with, 
built on [Restate](https://restate.dev). 

On top of a deep reseearch workflow, it has the following features:
- **Steer or cancel a run in flight.** A follow-up message either folds into the live run
  (*steer*) or cancels it — and its whole fan-out of researchers — and restarts on the new goal
  (*cancel*).
- **A governed LLM gateway.** Every workflow model call goes through one `LLMGateway` service that
  enforces a model allow-list and runs inside a concurrency-limited *scope*, so one CLI rule caps
  model spend across the whole department — no Redis, no semaphore.

The agent loop is hand-written on `litellm.acompletion`; durability, steering, session state, and
flow control all come from Restate. Delivery is Slack, or stdout when no Slack token is set.

## Architecture

| Object | Kind | Role |
|--------|------|------|
| `Controller` | Virtual Object | Entry point, keyed per session. Owns the chat history + the in-flight run's id; routes each new message to steer or cancel. |
| `DeepResearchAgent` | Virtual Object | The run: plan → approval → fan-out → writer, wrapped in a steer/cancel loop. |
| `ResearchAgent` | Service | One subtopic researcher (its own agent loop); N run in parallel. |
| `LLMGateway` | Service | Model allow-list policy, then the journaled model call — invoked through `scope("department1")` for flow control. |

(`DeepResearchAgentV1` is the same workflow at an earlier stage — local model calls, no gateway or
steering — kept side by side for comparison.)

## Run locally

### 1. Install and start the Restate server

```bash
docker run -p 8080:8080 -p 9070:9070 -p 9071:9071 \
  --add-host=host.docker.internal:host-gateway \
  -e RESTATE_EXPERIMENTAL_ENABLE_VQUEUES=true \
  -e RESTATE_EXPERIMENTAL_ENABLE_PROTOCOL_V7=true \
  docker.restate.dev/restatedev/restate:latest
```

### 2. Run the app — with or without API keys

**With live models + web search:**

```bash
uv sync
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
export SLACK_BOT_TOKEN=xoxb-...   # optional — without it, messages print to stdout
uv run app
```

**Offline — no keys, no network:** `OFFLINE=1` swaps in canned, schema-valid LLM and web-search
responses (including the researchers' tool calls), so the whole demo — fan-out, approval,
steer/cancel, flow control, retries — runs unchanged:

```bash
OFFLINE=1 uv run app
```

Then register the deployment: in the UI (`http://localhost:9070`) add
`http://host.docker.internal:9080`. You should see `Controller`, `DeepResearchAgent`,
`DeepResearchAgentV1`, `ResearchAgent`, and `LLMGateway`.

### 3. Durable execution

Every step of the run — plan, each researcher, each tool call, the writer — is journaled. Kill the
process mid-run and restart it: Restate replays the journal, so completed steps aren't repeated and
the run resumes exactly where it left off — no lost work, no duplicate LLM spend.

```bash
scripts/message.sh session55 "What's new in AI"
# while the fan-out is running, kill the app (Ctrl-C) and restart it:
OFFLINE=1 uv run app
```

Watch the Restate UI (`http://localhost:9070`): the invocation picks up from the last journaled
step. Bump `STUB_DELAY` to widen the window, and set `FAILURE_PROBABILITY` in `app/utils/tools.py`
to inject flaky web-search calls and watch Restate retry each to a journaled success.

### 4. Session coordination

Start a run through the session Controller, then interact with follow-ups — all through the same
`scripts/message.sh` (one endpoint; the Controller decides start vs. steer vs. cancel from its own
state). Check the UI to audit execution:

```bash
scripts/message.sh session55 "What's new in AI"                      
scripts/message.sh session55 "focus on frontier models"             
scripts/message.sh session55 "forget it, research AI policy instead" 
```

After planning, the run parks on the **human approval** gate. With Slack, click the **Approve**
button; in log mode there's no button, so the app log prints a copy-pasteable `curl` — paste it to
resolve the approval and let the run proceed (or send a follow-up to steer/cancel instead):

```bash
curl http://localhost:8080/restate/awakeables/<awk_id>/resolve --json '{"approved": true}'
```

### 5. Flow control

Cap concurrency on the scope (against your **local** server), then fire a swarm:

```bash
restate rules set department1 --concurrency 3 
scripts/swarm.sh 12                            
```

Watch the Restate UI (`http://localhost:9070`), or query `sys_vqueues` / `sys_user_limits`.

### Knobs

- **`STUB_DELAY`** (offline, default `2`s) — per-call delay so the parallel researchers and the
  steer/cancel window stay visible in the UI. `OFFLINE=1 STUB_DELAY=4 uv run app`.
- **`FAILURE_PROBABILITY`** in `app/utils/tools.py` — inject flaky web-search calls to watch
  Restate retry to a journaled success (works offline too).
- **Policy** — point a request at a model outside `APPROVED_MODELS` (`app/utils/config.py`) to see
  the gateway reject it with no model spend.

## Deploy

`deploy/modal_app.py` serves the same services on Modal for Restate Cloud. Serverless platforms
can't hold Restate's bidirectional stream, so it registers in **request/response** mode (not
`bidi`) — see the comments in that file.
