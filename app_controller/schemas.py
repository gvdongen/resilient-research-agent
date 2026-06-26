"""Plain pydantic models. `extra="forbid"` satisfies OpenAI strict json_schema."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrategyChoice(BaseModel):
    """How to handle a message that arrives while a run is already in flight."""
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["enqueue", "interrupt", "steer"]
    reason: str
