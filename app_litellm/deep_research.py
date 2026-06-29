"""Deep Research agent — the application.

A planner → parallel researchers → writer workflow, wrapped in a Controller that can
steer / interrupt / enqueue a run that is already in flight. Interrupt cancels the run
and its whole fan-out of researchers in one durable signal, then rolls forward.

The reusable engine — the LLM gateway (policy + flow control) and the durable tool
loop — lives in utils/agent.py. This file is just the workflow, the session state,
and the controller.
"""

import restate
from datetime import timedelta

from utils.agent import FAST_MODEL, TOOL_SPECS, durable_llm_call, run_agent
from utils.prompts import *
from utils.schemas import *
from utils.tools import post_plan, post_report, post_update, to_brief

CANCELLATION = 409


# ----------- Researcher — one subtopic, in parallel ---------------------

research_agent = restate.Service("ResearchAgent")


class Topic(BaseModel):
    session: str
    topic: str


@research_agent.handler()
async def investigate(ctx: restate.Context, topic: Topic) -> SubReport:
    return await run_agent(
        ctx,
        prompt=RESEARCHER_SYSTEM,
        messages=[{"role": "user", "content": f"Topic: {topic}"}],
        output=SubReport,
        tools=TOOL_SPECS,
    )


# ----------- The deep-research workflow ---------------------

deep_research_agent = restate.VirtualObject("DeepResearchAgent")


@deep_research_agent.handler()
async def research(ctx: restate.ObjectContext, history: ChatHistory):
    session = ctx.key()

    try:
        # 1 — plan
        plan: Plan = await run_agent(ctx, prompt=PLANNER, messages=history.messages, output=Plan)

        while True:
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
        report = await run_agent(ctx, prompt=WRITER, messages=[{"role": "user", "content": brief}], output=Report)
        await ctx.run_typed("slack-report", post_report, channel=session, report=report)

    except restate.TerminalError as e:
        if e.status_code == CANCELLATION:
            text="⏹️ Stopped this research — starting over on your new request."
            await ctx.run_typed("slack-update", post_update, channel=ctx.key(), text=text)
        raise
    ctx.object_send(done, key=ctx.key(), arg={"inv_id": ctx.request().id, "message": report.model_dump_json()})


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
        decision = await durable_llm_call(
            ctx, model=FAST_MODEL, instructions=CLASSIFIER, messages=[message], output=StrategyChoice,
        )
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
