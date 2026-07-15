#!/usr/bin/env bash
# Flow control across two scopes — fire N invocations at the LLMGateway under the
# "marketing" scope and M under "engineering", all at once.
#
# Each request goes STRAIGHT to LLMGateway/call_llm through the scoped ingress
# endpoint (/restate/scope/<scope>/send/...), so every invocation counts against
# that scope's concurrency budget — no workflow, no researchers, just the gateway.
# `/send` is fire-and-forget: each curl returns an invocation id immediately and
# the runs queue up behind the scope's concurrency limit.
#
# Set a limit per scope FIRST (against your LOCAL server), then run this:
#
#   restate rules set marketing   --concurrency 5     # local server (localhost:9070)
#   restate rules set engineering --concurrency 3
#   scripts/scope-swarm.sh
#
# Watch only <limit> call_llm invocations running per scope in the Restate UI
# (http://localhost:9070), or query the system tables:
#   SELECT * FROM sys_user_limits;   SELECT * FROM sys_vqueues;
#
# Local only: talks to http://localhost:8080, no auth.
#
# Usage: scripts/scope-swarm.sh [marketing_count] [engineering_count]
#   scripts/scope-swarm.sh              # 20 marketing + 15 engineering
#   scripts/scope-swarm.sh 40 30        # 40 + 30
set -euo pipefail

INGRESS="http://localhost:8080"
MARKETING_N="${1:-20}"
ENGINEERING_N="${2:-15}"

# Fire `count` gateway calls under `scope`, all at once (each curl backgrounded),
# so the whole batch lands together and queues behind the scope's concurrency limit.
fire_scope() {
  local scope="$1" count="$2" i body
  for i in $(seq 1 "$count"); do
    body=$(printf '{"msgs":[{"role":"user","content":"[%s] request %d of %d"}]}' "$scope" "$i" "$count")
    curl -sS "$INGRESS/restate/scope/$scope/send/LLMGateway/call_llm" \
      --json "$body" \
      -o /dev/null -w "  [$scope $i/$count] %{http_code}\n" &
  done
}

echo "Firing $MARKETING_N marketing + $ENGINEERING_N engineering LLMGateway calls at once at $INGRESS ..."
fire_scope marketing "$MARKETING_N"
fire_scope engineering "$ENGINEERING_N"
wait
echo "Done. Watch the per-scope queues in the Restate UI (http://localhost:9070) or sys_user_limits."
