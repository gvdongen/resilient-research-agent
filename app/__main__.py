"""Single entry point — serves every Restate service in one app.

Run with: `uv run app`
"""

import restate
from session_coordinator import controller
from deep_research import deep_research_agent, deep_research_agent_v2, research_agent
from llm_gateway import llm_gateway

app = restate.app(services=[controller, deep_research_agent, deep_research_agent_v2, research_agent, llm_gateway])


if __name__ == "__main__":
    import asyncio
    import hypercorn.asyncio
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:9080"]
    asyncio.run(hypercorn.asyncio.serve(app, config))
