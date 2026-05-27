"""Single entry point — serves every Restate service in one app.

Run with: `uv run python __main__.py`
"""

import restate
from deep_research import deep_research_agent, research_agent

app = restate.app([deep_research_agent, research_agent])


if __name__ == "__main__":
    import asyncio
    import hypercorn.asyncio
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:9080"]
    asyncio.run(hypercorn.asyncio.serve(app, config))
