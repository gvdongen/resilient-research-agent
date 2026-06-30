import restate as rst
from restate import Context

from app_litellm.utils.config import APPROVED_MODELS
from app_litellm.utils.schemas import LLMRequest
from app_litellm.utils.tools import provider_call


# ----------- LLM Gateway ---------------------

llm_gateway = rst.Service("LLMGateway")


@llm_gateway.handler()
async def call_llm(restate: Context, req: LLMRequest) -> dict:
    # 1. policy guardrail
    if req.model not in APPROVED_MODELS:
        raise rst.TerminalError(
            f"Model '{req.model}' is not on the approved list {sorted(APPROVED_MODELS)}"
        )

    # 2. call LLM
    response = await restate.run_typed("provider", provider_call, req=req)
    return response["choices"][0]["message"]

