"""Deep Research agent — the application.

A planner → parallel researchers → writer workflow, wrapped in a Controller that can
steer / interrupt / enqueue a run that is already in flight. Interrupt cancels the run
and its whole fan-out of researchers in one durable signal, then rolls forward.

The reusable engine — the LLM gateway (policy + flow control) and the durable tool
loop — lives in utils/agent.py. This file is just the workflow, the session state,
and the controller.
"""
import json
import restate
from datetime import timedelta
from utils.prompts import *
from utils.schemas import *
from utils.tools import *

MODEL = "gpt-5"
FAST_MODEL = "gpt-5-mini"
APPROVED_MODELS = {"gpt-5", "gpt-5-mini"}
CANCELLATION = 409
MAX_TURNS = 2

# ----------- LLM Gateway — policy + flow control ---------------------

llm_gateway = restate.Service("LLMGateway")


@llm_gateway.handler()
async def call_llm(ctx: restate.Context, req: LLMRequest) -> dict:
    # 1. policy guardrail
    if req.model not in APPROVED_MODELS:
        raise restate.TerminalError(
            f"Model '{req.model}' is not on the approved list {sorted(APPROVED_MODELS)}"
        )

    # 2. call LLM
    response = await ctx.run_typed("provider", provider_call, req=req)
    return response["choices"][0]["message"]


# ----------- Researcher — one subtopic, in parallel ---------------------


async def web_search_fn(query: str, time_range: Range = "month") -> dict:
    """Search the web. time_range is one of: day, week, month, year."""
    return call_websearch_api(query=query, range=time_range)

web_search = to_tool(web_search_fn)

async def extract_urls_fn(urls: list[str]) -> dict:
    """Return the full readable text of a list of web pages."""
    return await call_extract_api(urls=urls)

extract_urls = to_tool(extract_urls_fn)


async def crawl_site_fn(url: str, instructions: str = "") -> dict:
    """Crawl a website (guided by natural-language instructions) and return content from up to 10 pages."""
    return call_crawl_api(url=url, instructions=instructions)

crawl_site = to_tool(crawl_site_fn)

research_agent = restate.Service("ResearchAgent")


class Topic(BaseModel):
    session: str
    topic: str


@research_agent.handler()
async def investigate(ctx: restate.Context, topic: Topic) -> dict:
    msgs = [{"role": "user", "content": topic}]
    for _ in range(MAX_TURNS):
        investigation_request = LLMRequest(prompt=RESEARCHER, msgs=msgs, output=SubReport, tools=[web_search, extract_urls, crawl_site])
        response = await ctx.scope(topic.session).service_call(call_llm, arg=investigation_request)
        msgs.append(response)

        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            return response["content"]

        # Durable parallel tool calls
        handles = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            match name:
                case "web_search":
                    handles.append(ctx.run_typed(name, web_search_fn, **args))
                case "extract_urls":
                    handles.append(ctx.run_typed(name, extract_urls_fn, **args))
                case "crawl_site":
                    handles.append(ctx.run_typed(name, crawl_site_fn, **args))
        await restate.gather(*handles)
        for tc, h in zip(tool_calls, handles):
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(await h)})

    # Budget exhausted — force a final structured answer with no more tools
    msgs.append({"role": "user", "content": "Turn budget exhausted. Return the result now with what you have."})
    investigation_request = LLMRequest(prompt=RESEARCHER, msgs=msgs, output=SubReport)
    response = await ctx.scope(topic.session).service_call(call_llm, arg=investigation_request)
    return response["content"]


# ----------- The deep-research workflow ---------------------

deep_research_agent = restate.VirtualObject("DeepResearchAgent")


@deep_research_agent.handler()
async def research(ctx: restate.ObjectContext, history: ChatHistory):
    session = ctx.key()

    try:
        while True:
            # 1 — plan
            plan_request = LLMRequest(prompt=PLANNER, msgs=history.messages, output=Plan)
            plan = await ctx.scope(session).service_call(call_llm, arg=plan_request)

            # 2 — human approval: suspends with no compute held until the button is clicked
            awk_id, decision = ctx.awakeable(type_hint=Decision)
            await ctx.run_typed("slack-plan", post_plan, channel=session, plan=plan, awk_id=awk_id)
            if not (await decision).approved:
                return

            # 3 — fan out one researcher per subtopic. Cancelling this run cancels them all.
            handles = [ctx.service_call(investigate, arg=Topic(session=session, topic=sub)) for sub in plan.subtopics]
            await restate.gather(*handles)
            sub_reports = [await h for h in handles]

            # 4 — check for steering updates
            brief = to_brief(plan, sub_reports)
            match await restate.select(steer=ctx.signal("steer", type_hint=str), now=ctx.sleep(timedelta(0))):
                case ["steer", update]:
                    brief += f"\n# Steering update from the user — incorporate this\n{update}"
                case _:
                    break

        # 4 — synthesize and deliver
        write_request = LLMRequest(prompt=WRITER, msgs=[{"role": "user", "content": brief}], output=Report)
        report = await ctx.scope(session).service_call(call_llm, arg=write_request)

        await ctx.run_typed("slack-report", post_report, channel=session, report=report)

    except restate.TerminalError as e:
        if e.status_code == CANCELLATION:
            text="⏹️ Stopped this research — starting over on your new request."
            await ctx.run_typed("slack-update", post_update, channel=session, text=text)
        raise
    ctx.object_send(done, key=session, arg={"inv_id": ctx.request().id, "message": report.model_dump_json()})


# ----------- Controller — steer / interrupt / enqueue ---------------------

controller = restate.VirtualObject("Controller")

@controller.handler()
async def message(ctx: restate.ObjectContext, text: str) -> None:
    history = await ctx.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append({"role": "user", "content": text})
    ctx.set("messages", history)

    current = await ctx.get("current", type_hint=str)

    if current is not None:
        message={"role": "user", "content": f"Current goal: {history.messages[-3:]} - New message:\n{text}"}
        write_request = LLMRequest(model=FAST_MODEL, prompt=CLASSIFIER, msgs=[message], output=Strategy)
        decision = await ctx.scope(ctx.key()).service_call(call_llm, arg=write_request)

        match decision['strategy']:
            case "interrupt":
                ctx.cancel_invocation(current)  # cancels the run AND its whole fan-out
            case "steer":
                ctx.resolve_signal(current, "steer", text)
                return

    handle = ctx.object_send(research, key=ctx.key(), arg=history)
    ctx.set("current", await handle.invocation_id())


@controller.handler()
async def done(ctx: restate.ObjectContext, payload: dict) -> None:
    """Called by a run when it finishes. Record the reply."""
    history = await ctx.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append(payload["message"])
    ctx.set("messages", history)
    # clear the pointer if this is still the run we track (a newer enqueue/interrupt run may have replaced it(
    if await ctx.get("current", type_hint=str) == payload["inv_id"]:
        ctx.clear("current")
