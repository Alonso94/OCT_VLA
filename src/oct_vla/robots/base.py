"""Synchronous canonical robot contract; stop is latched until reset."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from oct_vla.core.action import Action
from oct_vla.core.observation import RobotObservation


@dataclass(frozen=True)
class StepResult:
    observation: RobotObservation
    applied_action: Action
    elapsed_seconds: float


@dataclass(frozen=True)
class Health:
    ready: bool
    detail: str


@runtime_checkable
class RobotBackend(Protocol):
    def reset(self, seed: int) -> RobotObservation: ...
    def observe(self) -> RobotObservation: ...
    def step(self, action: Action) -> StepResult: ...
    def stop(self) -> None: ...
    def health(self) -> Health: ...
    def close(self) -> None: ...
