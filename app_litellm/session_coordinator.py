import json

import restate as rst
from restate import ObjectContext

from app_litellm.llm_gateway import call_llm
from app_litellm.utils.config import FAST_MODEL
from app_litellm.utils.prompts import CLASSIFIER
from app_litellm.utils.schemas import ChatHistory, Strategy, LLMRequest
from app_litellm.utils.tools import post_to_slack

# ----------- Session Controller ---------------------

controller = rst.VirtualObject("Controller")

@controller.handler()
async def message(restate: ObjectContext, text: str) -> None:
    history = await restate.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append({"role": "user", "content": text})
    restate.set("messages", history)

    current = await restate.get("current", type_hint=str)

    if current is not None:
        message={"role": "user", "content": f"Current goal: {history.messages[-3:]} - New message:\n{text}"}
        write_request = LLMRequest(model=FAST_MODEL, prompt=CLASSIFIER, msgs=[message], output_schema=Strategy)
        decision = json.loads((await restate.scope(restate.key()).service_call(call_llm, arg=write_request))["content"])

        match decision['strategy']:
            case "cancel":
                restate.cancel_invocation(current)
            case _:
                restate.resolve_signal(current, "steer", text)
                return

    from deep_research import research
    handle = restate.object_send(research, key=restate.key(), arg=history)
    restate.set("current", await handle.invocation_id())














@controller.handler()
async def update_slack(restate: ObjectContext, msg: dict) -> None:
    history = await restate.get("messages", type_hint=ChatHistory) or ChatHistory()
    history.messages.append({"role": "assistant", "content": msg["text"]})
    restate.set("messages", history)

    await restate.run_typed("slack", post_to_slack, channel=restate.key(), text=msg["text"], awk_id=msg.get("awk_id"))

    if "inv_id" in msg and await restate.get("current", type_hint=str) == msg["inv_id"]:
        restate.clear("current")
