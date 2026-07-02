#
#  Copyright (c) 2023-2026 - Restate Software, Inc., Restate GmbH
#
#  This file is part of the Restate SDK for Python,
#  which is released under the MIT license.
#
#  You can find a copy of the license in file LICENSE in the root
#  directory of this repository or package, or at
#  https://github.com/restatedev/sdk-typescript/blob/main/LICENSE
#
"""LangChain agent middleware that makes a `create_agent` agent durable on Restate.

- `awrap_model_call` journals each LLM response so retries replay it from the
  journal instead of re-calling the model.
- `awrap_tool_call` runs parallel tool calls one at a time (via a turnstile
  keyed on `tool_call_id`) so any `ctx.run_typed(...)` calls users place in
  tool bodies appear in the journal in a stable order across replays.

The middleware does not journal tool calls itself and does not catch
exceptions — wrap side effects explicitly with `restate_context().run_typed(...)`
inside the tool body.
"""

import json
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Optional, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage, BaseMessage, convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import BaseModel

from restate import RunOptions
from restate.extensions import current_context
from restate.ext.turnstile import Turnstile

from restate.ext.langchain._state import get_or_create_state, state_from_ctx

from .config import DEPARTMENT
from .schemas import LLMRequest

ToolCallResult = ToolMessage | Command


class SerializableModelResponse(BaseModel):
    """Serializable mirror of `ModelResponse`.

    `result` uses `list[AnyMessage]` (a discriminated union)
    so AIMessage `tool_calls` survives serialization.
    `BaseMessage`, as on `ModelResponse`, would not.
    """

    result: list[AnyMessage]
    structured_response: Optional[Any] = None


class RestateMiddleware(AgentMiddleware):
    """Drop-in middleware that makes a `create_agent` agent durable on Restate.

    Pass it to `create_agent(..., middleware=[RestateMiddleware()])` and run
    the agent inside a Restate handler. LLM responses are journaled; parallel
    tool calls are linearized for deterministic replay.

    Args:
        run_options: forwarded to the LLM `ctx.run_typed` call (max attempts,
            retry intervals, ...). `serde` is set internally. Only used by the
            default journaling path.
        call_llm: when set, the model call is routed through the `LLMGateway`
            service handler inside `scope(DEPARTMENT)` — so it picks up the
            gateway's policy guardrail and flow control — instead of the
            middleware's own `ctx.run_typed`. When None (the default), the model
            call is journaled locally via `ctx.run_typed`.
    """

    def __init__(
        self,
        run_options: Optional[RunOptions[Any]] = None,
        call_llm: Optional[Any] = None,
    ):
        super().__init__()
        self._options: RunOptions[Any] = run_options or RunOptions()
        self._call_llm = call_llm

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        ctx = current_context()
        if ctx is None:
            raise RuntimeError(
                "RestateMiddleware must run inside a Restate handler. "
                "Call agent.ainvoke(...) from a handler that exposes a Restate Context."
            )

        if self._call_llm is not None:
            # Route the model call through the LLMGateway service so it inherits
            # the gateway's policy guardrail + flow control (scope(department)).
            journaled = await self._call_via_gateway(ctx, request)
        else:
            async def call_model() -> SerializableModelResponse:
                response = await handler(request)
                return SerializableModelResponse(**asdict(response))

            journaled = await ctx.run_typed("LLM call", call_model, self._options)

        # If the request asked for a Pydantic schema, restore the type.
        structured_response = journaled.structured_response
        schema = getattr(request.response_format, "schema", None)
        if structured_response is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
            structured_response = schema.model_validate(structured_response)

        # Install this turn's turnstile on ctx.extension_data so sibling
        # ``awrap_tool_call`` tasks (spawned by ``tool_node``'s gather) reach
        # the same object via the shared Restate ``Context`` — independent of
        # ContextVar inheritance. Mirrors how ``restate.ext.adk`` stores its
        # PluginState turnstile.
        ai_message = next((m for m in journaled.result if isinstance(m, AIMessage)), None)
        state = get_or_create_state(ctx)
        if ai_message is not None:
            tool_call_ids = [tid for tc in (ai_message.tool_calls or []) if (tid := tc.get("id")) is not None]
            state.turnstile = Turnstile(tool_call_ids)

        # Turn into ModelResponse as expected by the agent
        return ModelResponse(
            result=cast(list[BaseMessage], journaled.result),
            structured_response=structured_response,
        )

    async def _call_via_gateway(self, ctx: Any, request: ModelRequest) -> SerializableModelResponse:
        """Do the model call through the LLMGateway and reshape its reply into a
        SerializableModelResponse, so the rest of awrap_model_call is unchanged.

        The gateway speaks OpenAI/litellm dicts, so we translate the LangChain
        request (messages, tools, response schema) on the way in and rebuild an
        AIMessage on the way out."""
        msgs = convert_to_openai_messages(request.messages)
        if request.system_message is not None:
            content = getattr(request.system_message, "content", request.system_message)
            msgs = [{"role": "system", "content": content}, *msgs]

        tools = [convert_to_openai_tool(t) for t in request.tools] if request.tools else None
        schema = getattr(request.response_format, "schema", None)  # a pydantic class; LLMRequest coerces it

        # A LangChain chat model exposes its id as `model_name`/`model`; `.name` is
        # the (usually None) Runnable name. Fall back to LLMRequest's default model.
        model = next(
            (m for m in (getattr(request.model, a, None) for a in ("model_name", "model", "name")) if isinstance(m, str) and m),
            None,
        )
        llm_request = LLMRequest(
            msgs=msgs, tools=tools, output_schema=schema, **({"model": model} if model else {})
        )

        # service_call is itself journaled, so no ctx.run_typed wrapper is needed.
        # Returns response["choices"][0]["message"] — the assistant message dict.
        message = await ctx.scope(DEPARTMENT).service_call(self._call_llm, arg=llm_request)

        ai_message = AIMessage(
            content=message.get("content") or "",
            id=str(ctx.uuid()),
            tool_calls=[
                {
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"],
                    "id": tc["id"],
                    "type": "tool_call",
                }
                for tc in (message.get("tool_calls") or [])
            ],
        )

        # Structured output only on the final turn: content is the schema JSON and
        # there are no tool calls. Leave it as a dict — awrap_model_call validates
        # it against the schema, mirroring the default path.
        structured_response = None
        if schema is not None and message.get("content") and not message.get("tool_calls"):
            structured_response = json.loads(message["content"])

        return SerializableModelResponse(result=[ai_message], structured_response=structured_response)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolCallResult]],
    ) -> ToolCallResult:
        tool_call_id = request.tool_call.get("id")
        if tool_call_id is None:
            return ToolMessage("Function call ID is required for tool invocation. Please set `id` on the ToolCall.")

        ctx = current_context()
        assert ctx is not None, "RestateMiddleware must run inside a Restate handler"
        state = state_from_ctx(ctx)
        assert state is not None, "RestateMiddleware must run inside a Restate handler"
        turnstile = state.turnstile

        try:
            await turnstile.wait_for(tool_call_id)
            result = await handler(request)
            turnstile.allow_next_after(tool_call_id)
            if isinstance(result, ToolMessage):
                result.id = str(ctx.uuid())
            return result
        except BaseException:
            # Unblock the rest of the parallel tool batch, then propagate.
            turnstile.cancel_all_after(tool_call_id)
            raise
