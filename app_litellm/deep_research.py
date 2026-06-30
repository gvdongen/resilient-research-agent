import json

import restate as rst
from restate import ObjectContext, Context
from langchain.agents import create_agent
from langchain_core.tools import tool
from restate.ext.langchain import restate_context
from utils.middleware import RestateMiddleware

from utils.prompts import *
from utils.schemas import *
from utils.tools import *

MODEL = "gpt-4o-mini"
FAST_MODEL = "gpt-4o-mini"
APPROVED_MODELS = {"gpt-5", "gpt-5-mini", "gpt-4o-mini", "gpt-4o"}
CANCELLATION = 409
department = "billing"


# ----------- LLM Gateway — policy + flow control ---------------------

llm_gateway = rst.Service("LLMGateway")


@llm_gateway.handler()
async def call_llm(ctx: rst.Context, req: LLMRequest) -> dict:
    # 1. policy guardrail
    if req.model not in APPROVED_MODELS:
        raise rst.TerminalError(
            f"Model '{req.model}' is not on the approved list {sorted(APPROVED_MODELS)}"
        )

    # 2. call LLM
    response = await ctx.run_typed("provider", provider_call, req=req)
    return response["choices"][0]["message"]


# ----------- Researcher — one subtopic, in parallel ---------------------


# Tools the researcher can call. Each wraps a Tavily call in a durable
# ctx.run_typed via restate_context(), so retries/replays are journaled.

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
    middleware=[
        RestateMiddleware(call_llm=call_llm, department=department, model=MODEL),
    ],
)

research_agent = rst.Service("ResearchAgent")


@research_agent.handler()
async def investigate(ctx: Context, topic: str) -> dict:
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
        plan = json.loads((await restate.scope(department).service_call(call_llm, arg=plan_request))["content"])

        sub_reports = []
        if plan["subtopics"]:
            # 2 - human approval
            awk_id, decision = restate.awakeable(type_hint=Decision)
            restate.object_send(update_slack, key=session, arg={"text": format_plan(plan), "awk_id": awk_id})
            if not (await decision).approved:
                return

            # 3 — research
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
        draft = await restate.scope(department).service_call(call_llm, arg=write_request)
        report = format_report(draft)
        break



    restate.object_send(update_slack, key=session, arg={"text": report, "inv_id": restate.request().id})


# ----------- Controller — steer / interrupt / enqueue ---------------------

controller = rst.VirtualObject("Controller")

@controller.handler()
async def message(ctx: rst.ObjectContext, text: str) -> None:
    history = await ctx.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append({"role": "user", "content": text})
    ctx.set("messages", history)

    current = await ctx.get("current", type_hint=str)

    if current is not None:
        message={"role": "user", "content": f"Current goal: {history.messages[-3:]} - New message:\n{text}"}
        write_request = LLMRequest(model=FAST_MODEL, prompt=CLASSIFIER, msgs=[message], output_schema=Strategy)
        decision = json.loads((await ctx.scope(ctx.key()).service_call(call_llm, arg=write_request))["content"])

        match decision['strategy']:
            case "cancel":
                ctx.cancel_invocation(current)
            case _:
                ctx.resolve_signal(current, "steer", text)
                return

    handle = ctx.object_send(research, key=ctx.key(), arg=history)
    ctx.set("current", await handle.invocation_id())


@controller.handler()
async def update_slack(ctx: rst.ObjectContext, msg: dict) -> None:
    history = await ctx.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append({"role": "assistant", "content": msg["text"]})
    ctx.set("messages", history)

    await ctx.run_typed("slack", post_to_slack, channel=ctx.key(), text=msg["text"], awk_id=msg.get("awk_id"))

    if "inv_id" in msg and await ctx.get("current", type_hint=str) == msg["inv_id"]:
        ctx.clear("current")
