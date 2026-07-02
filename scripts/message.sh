#!/usr/bin/env bash
# Local demo — send a message to a session Controller.
#
# There is one endpoint; the Controller decides what to do from its own state:
#   - no run in flight  -> start a new research run
#   - run in flight      -> classify the message and steer or cancel it
# Offline the classifier is keyword-based:
#   "forget / instead / stop / cancel / different"  -> cancel + restart
#   anything else                                    -> steer (fold into the live run)
#
# When a run reaches the plan-approval step, the app log (`uv run app` /
# `OFFLINE=1 uv run app`) prints how to approve the plan, or you can send a
# follow-up here to steer/cancel instead.
#
# Local only: talks to http://localhost:8080 (override with INGRESS=...).
#
# Usage: scripts/message.sh [session] [message...]
#   scripts/message.sh                                     # session "demo", "What's new in AI"
#   scripts/message.sh demo "focus on frontier models"     # steer a live run
#   scripts/message.sh demo forget it, research AI policy  # cancel + restart
set -euo pipefail

SESSION="${1:-demo}"
shift || true
MESSAGE="${*:-What's new in AI}"
INGRESS="${INGRESS:-http://localhost:8080}"

echo "[$SESSION] sending: $MESSAGE"
curl -sS "$INGRESS/Controller/$SESSION/message" \
  --json "$(printf '"%s"' "$MESSAGE")" \
  -o /dev/null -w "  sent (%{http_code})\n"

echo "Watch the app log for the plan + approve command, and the Restate UI (http://localhost:9070)."
echo "Steer it while it waits:  scripts/message.sh $SESSION \"focus on frontier models\""
