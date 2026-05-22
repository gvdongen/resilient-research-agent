"""`init_restate_model` — `init_chat_model` with durable invoke.

Each `ainvoke` on the returned model is wrapped in `ctx.run_typed`, so
retries replay the LLM response from the journal instead of re-calling
the model. Streaming is disabled — there is nothing to journal mid-stream.

Usage:

    from langchain.agents import create_agent
    from utils.restate_chat_model import init_restate_model

    model = init_restate_model("openai:gpt-5")
    agent = create_agent(model=model, tools=[...])

    # inside a Restate handler:
    result = await agent.ainvoke({"messages": ...})
"""

from typing import Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import PrivateAttr

from restate import RunOptions
from restate.ext.langchain import restate_context


class RestateChatModel(BaseChatModel):
    """Wrap any LangChain chat-model `Runnable` so `ainvoke` is journaled.

    Inherits `BaseChatModel` so IDEs and `create_agent`'s `isinstance` checks
    see a chat-model surface. Overrides `ainvoke` to wrap the inner model's
    call in `ctx.run_typed`. `invoke`, `stream`, and `astream` raise —
    Restate handlers are async and journal whole responses, not chunks.
    """

    _inner: Any = PrivateAttr()
    _journal_name: str = PrivateAttr()
    _options: RunOptions[AIMessage] = PrivateAttr()

    def __init__(self, inner: Any, *, journal_name: str = "LLM call",
                 run_options: Optional[RunOptions[AIMessage]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._inner = inner
        self._journal_name = journal_name
        self._options = run_options or RunOptions(type_hint=AIMessage)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        raise NotImplementedError(
            "RestateChatModel only supports ainvoke — Restate handlers are async."
        )

    @property
    def _llm_type(self) -> str:
        return self._inner._llm_type

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        ctx = restate_context()

        async def _call() -> AIMessage:
            return await self._inner.ainvoke(input, config=config, **kwargs)

        return await ctx.run_typed(self._journal_name, _call, self._options)

    def invoke(self, *_args: Any, **_kwargs: Any) -> AIMessage:
        raise NotImplementedError(
            "RestateChatModel only supports ainvoke — Restate handlers are async."
        )

    def stream(self, *_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError("RestateChatModel does not support streaming.")

    async def astream(self, *_args: Any, **_kwargs: Any):
        raise NotImplementedError("RestateChatModel does not support streaming.")
        if False:  # makes this function an async generator
            yield

    def _rewrap(self, inner: Any) -> "RestateChatModel":
        return RestateChatModel(
            inner, journal_name=self._journal_name, run_options=self._options
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> "RestateChatModel":
        return self._rewrap(self._inner.bind_tools(tools, **kwargs))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "RestateChatModel":
        return self._rewrap(self._inner.with_structured_output(schema, **kwargs))

    def bind(self, **kwargs: Any) -> "RestateChatModel":
        return self._rewrap(self._inner.bind(**kwargs))


def init_durable_model(*args: Any, **kwargs: Any) -> RestateChatModel:
    """Same args as `langchain.chat_models.init_chat_model`, plus durable invoke.

    Returns a model where each `ainvoke` call is journaled via
    `ctx.run_typed`. Streaming is disabled.
    """
    return RestateChatModel(init_chat_model(*args, **kwargs))
