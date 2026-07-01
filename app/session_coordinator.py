import restate as rst
from restate import ObjectContext

from llm_gateway import call_llm_gateway
from utils.schemas import ChatHistory
from utils.tools import post_to_slack, user, assistant, classify_request

# ----------- Session Controller ---------------------

controller = rst.VirtualObject("Controller")







# STATEFUL ACTORS & SESSION COORDINATION & CONTROL

@controller.handler()
async def message(restate: ObjectContext, text: str) -> None:
    # 1. Add to context
    history = await restate.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append(user(text))
    restate.set("messages", history)

    # 2. Check for ongoing research
    current = await restate.get("current", type_hint=str)

    # 3a. Cancel or steer ongoing research
    if current is not None:
        decision = await call_llm_gateway(restate, classify_request(history, text))
        match decision["strategy"]:
            case "cancel":
                restate.cancel_invocation(current)
            case _:
                restate.resolve_signal(current, "steer", text)
                return

    # 3b. Start new research
    from deep_research import research
    handle = restate.object_send(research, key=restate.key(), arg=history)
    restate.set("current", await handle.invocation_id())














@controller.handler()
async def update_slack(restate: ObjectContext, msg: dict) -> None:
    history = await restate.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append(assistant(msg["text"]))
    restate.set("messages", history)

    await restate.run_typed("slack", post_to_slack, channel=restate.key(), text=msg["text"], awk_id=msg.get("awk_id"))

    if "inv_id" in msg and await restate.get("current", type_hint=str) == msg["inv_id"]:
        restate.clear("current")
