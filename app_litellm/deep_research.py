import restate as rst
from restate import ObjectContext, Context
from langchain.agents import create_agent
from langchain_core.tools import tool
from restate.ext.langchain import restate_context

from app_controller.llm import MODEL
from llm_gateway import call_llm
from session_coordinator import update_slack
from utils.config import DEPARTMENT
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












# ----------- The deep-research workflow ---------------------

deep_research_agent = rst.VirtualObject("DeepResearchAgent")


@deep_research_agent.handler()
async def research(restate: ObjectContext, history: ChatHistory):
    session = restate.key()

    while True:
        # 1 — plan
        plan_request = LLMRequest(prompt=PLANNER, msgs=history.messages, output_schema=Plan)
        plan = json.loads((await restate.scope(DEPARTMENT).service_call(call_llm, arg=plan_request))["content"])

        # 2 — research
        sub_reports = []
        if plan["subtopics"]:
            # 2a - human approval
            awk_id, decision = restate.awakeable(type_hint=Decision)
            restate.object_send(update_slack, key=session, arg={"text": format_plan(plan), "awk_id": awk_id})
            if not (await decision).approved:
                return

            # 2b - subtopic research
            handles = [restate.service_call(investigate, arg=topic) for topic in plan["subtopics"]]
            await rst.gather(*handles)
            sub_reports = [await h for h in handles]  # keep findings across steers

        # 4 - steer
        if text := await peek(restate.signal("steer", type_hint=str)):
            append(history, plan, sub_reports, text)
            continue

        # 5 — write
        brief = to_brief(plan, sub_reports)
        write_request = LLMRequest(prompt=WRITER, msgs=history.messages + [{"role": "user", "content": brief}], output_schema=Report)
        draft = await restate.scope(DEPARTMENT).service_call(call_llm, arg=write_request)
        report = format_report(draft)
        break

    restate.object_send(update_slack, key=session, arg={"text": report, "inv_id": restate.request().id})
