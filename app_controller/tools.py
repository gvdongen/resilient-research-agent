"""The agent's single tool: web search. Kept tiny."""

import json
import random
import time

from tavily import TavilyClient

# Bump to e.g. 0.2 to demo transient failures + Restate retries.
FAILURE_PROBABILITY = 0.0

tavily = TavilyClient()


def websearch(arguments: str) -> dict:
    """Search the web. `arguments` is the raw JSON the model produced for the call,
    e.g. {"query": "...", "time_range": "day|week|month|year"}."""
    args = json.loads(arguments)
    if random.random() < FAILURE_PROBABILITY:
        time.sleep(random.uniform(0, 2))
        raise Exception("Tavily API down")
    return tavily.search(
        query=args["query"],
        time_range=args.get("time_range", "month"),
        search_depth="advanced",
    )


# OpenAI tool spec + dispatch table for the agent loop.
websearch_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web. time_range is one of day, week, month, year.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "time_range": {"type": "string", "enum": ["day", "week", "month", "year"]},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
