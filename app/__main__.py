"""Single entry point — serves every Restate service in one app.

Run with: `uv run python __main__.py`
"""

import asyncio

import restate

from a_researcher import agent as resilient_agent
from b_research_session import agent as research_session
from c_deep_research import (
    agent as orchestrator_agent,
    news_scout_agent,
    planner_agent,
    research_agent,
    writer_agent,
)

app = restate.app(
    [
        resilient_agent,  # Phase 1
        research_session,  # Phase 2
        orchestrator_agent,  # Phase 3
        news_scout_agent,
        planner_agent,
        research_agent,
        writer_agent,
    ]
)


if __name__ == "__main__":
    import hypercorn.asyncio
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:9080"]
    asyncio.run(hypercorn.asyncio.serve(app, config))
