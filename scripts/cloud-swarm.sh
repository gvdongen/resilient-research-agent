#!/usr/bin/env bash
# Kick off N deep-research runs on Restate Cloud — one per session key.
#
# Sends each request straight to the DeepResearchAgent virtual object's `research`
# handler (bypassing the Controller), using the `/send` suffix so it's
# fire-and-forget: all N runs start concurrently and each curl returns an
# invocation id immediately instead of blocking on the (long, approval-gated) run.
#
# The `research` handler takes a ChatHistory, so the body is
# {"messages":[{"role":"user","content":"<prompt>"}]} — not a bare string.
#
# Requires:
#   export RESTATE_INGRESS=https://<env>.env.<region>.restate.cloud:8080
#   export RESTATE_AUTH_TOKEN=<your-restate-api-key>
#
# Usage: scripts/cloud-swarm.sh [N] [prompt]
#   scripts/cloud-swarm.sh                       # 10 runs, default prompt
#   scripts/cloud-swarm.sh 25 "What is new in durable execution?"
set -euo pipefail

# A .env file is NOT auto-loaded by bash, so a child process (this script) won't
# see vars you only put there. Load the repo-root .env when the vars aren't
# already exported in your shell (explicit exports win over the file).
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/.env"
if [ -z "${RESTATE_INGRESS:-}" ] && [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
fi

: "${RESTATE_INGRESS:?set RESTATE_INGRESS (env or .env) to your Restate Cloud ingress URL}"
: "${RESTATE_AUTH_TOKEN:?set RESTATE_AUTH_TOKEN (env or .env) to your Restate Cloud API key}"

N="${1:-10}"
PROMPT="${2:-What is new in AI agents?}"
RUN="swarm-$(date +%s)"   # unique key prefix so re-runs start fresh VOs instead of queueing

body=$(printf '{"messages":[{"role":"user","content":"%s"}]}' "$PROMPT")

echo "Firing $N deep-research runs at $RESTATE_INGRESS (prefix $RUN) ..."
for i in $(seq 1 "$N"); do
  key="$RUN-$i"
  curl -sS "$RESTATE_INGRESS/DeepResearchAgent/$key/research/send" \
    -H "Authorization: Bearer $RESTATE_AUTH_TOKEN" \
    --json "$body" \
    -o /dev/null -w "  [$key] %{http_code}\n" &
done
wait
echo "Done. Watch the runs in the Restate UI."
