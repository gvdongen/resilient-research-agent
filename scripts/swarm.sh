#!/usr/bin/env bash
# Flow-control demo: fire N research runs across N separate sessions at once.
#
# Each run's planner + parallel researchers + writer all call the LLMGateway
# through the "llm" scope, so with `restate rules set llm --concurrency 3` you
# see only 3 LLMGateway/complete invocations running at a time and the rest
# queued (watch the Restate UI, or query sys_vqueues / sys_user_limits).
#
# Usage: scripts/swarm.sh [N] [ingress-url] [prompt]
set -euo pipefail

N="${1:-12}"
INGRESS="${2:-http://localhost:8080}"
PROMPT="${3:-What is new in AI agents?}"

echo "Firing $N runs at $INGRESS ..."
for i in $(seq 1 "$N"); do
  curl -s "$INGRESS/Controller/sess-$i/message" --json "\"$PROMPT\"" >/dev/null &
done
wait
echo "Done. Watch the LLMGateway queue in the Restate UI (http://localhost:9070)."
