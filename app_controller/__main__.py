"""Single entry point — serves every service for the controller demo.

Run with: `uv run python app_controller/__main__.py`
"""

import restate

from controller import controller
from research import research_agent

app = restate.app([research_agent, controller])


if __name__ == "__main__":
    import asyncio

    import hypercorn.asyncio
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:9080"]
    asyncio.run(hypercorn.asyncio.serve(app, config))
