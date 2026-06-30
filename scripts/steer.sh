#!/usr/bin/env bash
# Steering demo: send two messages to the SAME session, 5 seconds apart.
#
# The first kicks off a run; the second arrives mid-flight, so the Controller
# classifies it and folds it in (steer) or cancels and restarts (interrupt).
#
# Usage: scripts/steer.sh [session] [ingress-url]
set -euo pipefail

SESSION="${1:-sess-$RANDOM}"
INGRESS="${2:-http://localhost:8080}"

echo "[$SESSION] sending: What is AI"
curl -s "$INGRESS/Controller/$SESSION/message" --json '"What is AI"' >/dev/null

sleep 5

echo "[$SESSION] sending: Answer as a poem"
curl -s "$INGRESS/Controller/$SESSION/message" --json '"Answer as a poem"' >/dev/null

echo "Done. Watch the run in the Restate UI (http://localhost:9070)."
