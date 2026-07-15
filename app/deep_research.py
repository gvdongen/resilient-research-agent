import restate as rst
from restate import ObjectContext, Context
from langchain.agents import create_agent
from langchain_core.tools import tool
from restate.ext.langchain import restate_context

from llm_gateway import call_llm_gateway
from session_coordinator import update_slack
from utils.config import MODEL
from utils.middleware import RestateMiddleware
from utils.prompts import *
from utils.schemas import *
from utils.tools import *


# ----------- Researcher — one subtopic, in parallel ---------------------


@tool
async def web_search(queries: list[str], time_range: Range = "month") -> list[str]:
    """Search the web with a list of queries. time_range is one of: day, week, month, year."""
    searches = [
        restate_context().run_typed(f"web_search:{q}", call_websearch_api, query=q, range=time_range)
        for q in queries
    ]
    await rst.gather(*searches)
    return [await s for s in searches]


researcher = create_agent(
    model="openai:" + MODEL,
    tools=[web_search],
    system_prompt=RESEARCHER,
    response_format=SubReport,
    middleware=[RestateMiddleware()],
)

research_agent = rst.Service("ResearchAgent")


@research_agent.handler()
async def investigate(_restate: Context, topic: str) -> dict:
    result = await researcher.ainvoke({"messages": f"Topic: {topic}"})
    return result["structured_response"].model_dump()


# ----------- Stage 1 — deep research v1 ---------

deep_research_agent = rst.VirtualObject("DeepResearchAgentV1")


# DURABLE EXECUTION FOR LONG-RUNNING AGENTS

@deep_research_agent.handler()
async def deep_research(restate: ObjectContext, history: ChatHistory):
    session = restate.key()

    # 1. plan
    plan = await restate.run_typed("plan", llm_call, req=plan_request(history))

    # 2. human approval
    awk_id, decision = restate.awakeable(type_hint=Decision)
    restate.object_send(update_slack, key=session, arg=format_plan(plan, awk_id))
    if not (await decision).approved:
        return

    # 3. parallel subagent research
    handles = [restate.service_call(investigate, arg=topic) for topic in plan["subtopics"]]
    await rst.gather(*handles)
    sub_reports = [await h for h in handles]

    # 4. write
    draft = await restate.run_typed("write", llm_call, req=write_request(history, plan, sub_reports))

    # 5. Slack message
    restate.object_send(update_slack, key=session, arg=format_report(draft, restate.request().id))


# ----------- Stage 2 — deep research v2 -------------------

deep_research_agent_v2 = rst.VirtualObject("DeepResearchAgent")


@deep_research_agent_v2.handler()
async def research(restate: ObjectContext, history: ChatHistory):
    session = restate.key()

    while True:
        # one steer slot per round — raced against approval, then re-checked before writing
        steer = restate.signal("steer", type_hint=str)

        # 1 — plan
        plan = await call_llm_gateway(restate, plan_request(history))

        # 2 — research
        sub_reports = []
        if plan["subtopics"]:
            # 2a - human approval — race the Approve button against a steer message
            awk_id, decision = restate.awakeable(type_hint=Decision)
            restate.object_send(update_slack, key=session, arg=format_plan(plan, awk_id))
            match await rst.select(approval=decision, steer=steer):
                case ["steer", text]:
                    append(history, plan, sub_reports, text)
                    continue
                case _:
                    pass

            # 2b - subtopic research
            handles = [restate.service_call(investigate, arg=topic) for topic in plan["subtopics"]]
            await rst.gather(*handles)
            sub_reports = [await h for h in handles]  # keep findings across steers

        # 4 - steer (arrived while researching)
        if text := await peek(steer):
            append(history, plan, sub_reports, text)
            continue

        # 5 — write
        draft = await call_llm_gateway(restate, write_request(history, plan, sub_reports))
        break

    restate.object_send(update_slack, key=session, arg=format_report(draft, restate.request().id))
