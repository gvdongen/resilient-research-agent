#!/usr/bin/env bash
# Local demo (part 3) — flow control: fire N research runs across N sessions at once.
#
# Each run's planner + writer (and, offline, the researchers) call the LLMGateway
# through the "department1" scope. Set a concurrency limit on that scope FIRST —
# against your LOCAL server — then run this:
#
#   restate rules set department1 --concurrency 3     # local server (localhost:9070)
#   scripts/swarm.sh 12
#
# You'll see only 3 LLMGateway/call_llm invocations running at a time and the rest
# queued — watch the Restate UI (http://localhost:9070) or query the system tables:
#   SELECT * FROM sys_vqueues;   SELECT * FROM sys_user_limits;
#
# Local only: talks to http://localhost:8080 (override with arg 2).
#
# Usage: scripts/swarm.sh [N] [ingress-url] [prompt]
set -euo pipefail

DEFAULT_PROMPT="What's new in AI"
N="${1:-12}"
INGRESS="${2:-http://localhost:8080}"
PROMPT="${3:-$DEFAULT_PROMPT}"

echo "Firing $N runs at $INGRESS ..."
for i in $(seq 1 "$N"); do
  curl -s "$INGRESS/Controller/sess-$i/message" --json "$(printf '"%s"' "$PROMPT")" >/dev/null &
done
wait
echo "Done. Watch the LLMGateway queue in the Restate UI (http://localhost:9070)."
