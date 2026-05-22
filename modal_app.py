"""Deploy the Restate services from `app/` to Modal as one ASGI endpoint.

Register the resulting URL with Restate Cloud (or any Restate server) as a
deployment. Restate then calls back into this endpoint over HTTP/2 to drive
every handler.

    modal deploy modal_app.py
    restate dp register https://<org>--deep-research-agent-restate-services.modal.run
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.14")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("app", remote_path="/root/app")
)

app = modal.App("deep-research-agent", image=image)


@app.function(
    secrets=[modal.Secret.from_name("research-agent-secrets")],
    # min_containers=1,
    timeout=36000,
)
@modal.asgi_app()
def restate_services():
    import sys

    sys.path.insert(0, "/root/app")

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

    return restate.app(
        [
            resilient_agent,
            research_session,
            orchestrator_agent,
            news_scout_agent,
            planner_agent,
            research_agent,
            writer_agent,
        ]
    )
