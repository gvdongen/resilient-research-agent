"""Raw-HTTP LLM client — no OpenAI SDK, no litellm, no LangChain.

Just `httpx` posting to the OpenAI chat-completions endpoint. The point of the
demo is that durability comes entirely from Restate (every call here is wrapped
in `ctx.run_typed(...)` by the callers), not from any agent framework.

Three entry points:
- `chat_text`       — plain text answer.
- `chat_structured` — structured output via `response_format` json_schema (the classifier).
- `call_llm`        — one agent turn: returns an `LLMResponse` with the text and any
                      tool calls already parsed out, so the agent loop stays clean.
"""

import json
import os

import httpx
from pydantic import BaseModel


class ToolCall(BaseModel):
    """One tool call the model wants to make. `arguments` is the raw JSON string the
    model produced — we hand it to the tool untouched and let the tool parse it."""
    id: str
    name: str
    arguments: str


class LLMResponse(BaseModel):
    """One assistant turn, parsed into the bits the agent loop actually uses."""
    message: dict             # raw assistant message — appended to history verbatim
    response: str             # the text answer ("" while the model is still tool-calling)
    tool_calls: list[ToolCall]


# Chat-message constructors — so the agent loop reads in intent, not dict literals.
def system(content: str) -> dict:
    return {"role": "system", "content": content}


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def tool_result(call: ToolCall, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call.id, "content": content}

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Main model for planning / research / writing. FAST_MODEL is used for the
# cheap controller classifier. Change here if these names ever drift.
MODEL = "gpt-5"
FAST_MODEL = "gpt-5-mini"


async def _post(payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(OPENAI_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def _json_schema_format(model: type[BaseModel]) -> dict:
    """OpenAI strict structured-output format from a pydantic model.

    Our models use `ConfigDict(extra="forbid")` (-> additionalProperties: false)
    and have no defaulted fields, which is what strict json_schema requires."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": model.model_json_schema(),
        },
    }


async def chat_text(messages: list[dict], model: str = MODEL) -> str:
    """Plain text completion."""
    data = await _post({"model": model, "messages": messages})
    return data["choices"][0]["message"]["content"]


async def chat_structured(
    messages: list[dict], schema_model: type[BaseModel], model: str = MODEL
) -> dict:
    """Structured completion. Returns a plain dict; caller validates into `schema_model`."""
    data = await _post(
        {
            "model": model,
            "messages": messages,
            "response_format": _json_schema_format(schema_model),
        }
    )
    return json.loads(data["choices"][0]["message"]["content"])


async def call_llm(messages: list[dict], tools: list[dict], model: str = MODEL) -> LLMResponse:
    """One agent turn with tools available."""
    data = await _post({"model": model, "messages": messages, "tools": tools})
    msg = data["choices"][0]["message"]
    return LLMResponse(
        message=msg,
        response=msg.get("content") or "",
        tool_calls=[
            ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"])
            for tc in msg.get("tool_calls") or []
        ],
    )
