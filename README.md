# A steerable, governed research agent on Restate

A long-running deep-research agent — **plan → human approval → parallel researchers → writer** —
that you can **talk to while it works**, built on [Restate](https://restate.dev). Two things make
it more than a durable workflow:

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

Flow control (scoped concurrency) needs a recent server with virtual queues enabled:

```bash
uv sync
docker pull docker.restate.dev/restatedev/restate:latest
docker run -p 8080:8080 -p 9070:9070 -p 9071:9071 \
  --add-host=host.docker.internal:host-gateway \
  -e RESTATE_EXPERIMENTAL_ENABLE_VQUEUES=true \
  docker.restate.dev/restatedev/restate:latest
```

### 2. Run the app — with or without API keys

**With live models + web search:**

```bash
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

### 3. Drive it

```bash
# start a run for session "demo"
curl localhost:8080/Controller/demo/message --json '"What is new in AI agents?"'

# approve the plan (copy the awakeable curl printed in the app logs):
curl localhost:8080/restate/awakeables/<awk_id>/resolve --json '{"approved": true}'

# while it runs, send a follow-up — classified as steer or cancel:
curl localhost:8080/Controller/demo/message --json '"also cover pricing"'              # steer
curl localhost:8080/Controller/demo/message --json '"forget it — research X instead"'  # cancel
```

Offline, the classifier is keyword-based so each path is reproducible:
*forget / instead / stop / cancel / different* → **cancel**, anything else → **steer**.

### 4. Demo the flow control

```bash
restate rules set department1 --concurrency 3   # cap concurrent model calls for the scope
scripts/swarm.sh 12                             # fire 12 runs; only 3 LLMGateway calls run at once
```

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
