"""Single entry point — serves every Restate service in one app_litellm.

Run with: `uv run .`
"""

import asyncio

import restate

from resilient_agent import agent as resilient_agent
from stateful_agent import agent as stateful_agent
from deep_research import agent as orchestrator_agent, news_scout_agent, planner_agent, research_agent, writer_agent


app = restate.app([
    resilient_agent,
    stateful_agent,
    orchestrator_agent,
    news_scout_agent,
    planner_agent,
    research_agent,
    writer_agent,
])

if __name__ == "__main__":
    import hypercorn.asyncio
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:9080"]
    asyncio.run(hypercorn.asyncio.serve(app, config))
