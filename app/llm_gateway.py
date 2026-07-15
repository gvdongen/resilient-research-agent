import restate as rst
from restate import Context, TerminalError

from utils.config import APPROVED_MODELS, DEPARTMENT
from utils.schemas import LLMRequest
from utils.tools import llm_call

llm_gateway = rst.Service("LLMGateway")


# FLEXIBILITY & DISTRIBUTED COMMUNICATION & FLOW CONTROL

@llm_gateway.handler()
async def call_llm(restate: Context, req: LLMRequest) -> dict:
    # 1. policy guardrail
    if req.model not in APPROVED_MODELS:
        raise TerminalError(f"Model '{req.model}' is not on the approved.")

    # 2. call LLM
    return await restate.run_typed("llm", llm_call, req=req)


# ----------- Flow control ---------------------
# Cap concurrent model calls across the whole department with one CLI rule:
#   restate rules set department1 --concurrency 300


# --------- Call to LLM Gateway ---------------------

async def call_llm_gateway(restate: Context, req: LLMRequest) -> dict:
    return await restate.scope(DEPARTMENT).service_call(call_llm, arg=req)
