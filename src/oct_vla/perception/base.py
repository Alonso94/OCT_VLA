"""Canonical object-perception contract; ground-truth and vision share this output."""

from typing import Protocol, runtime_checkable

from oct_vla.core.objects import ObjectScene
from oct_vla.core.observation import RobotObservation


@runtime_checkable
class ObjectStateEstimator(Protocol):
    def estimate(self, observation: RobotObservation) -> ObjectScene: ...
