"""Canonical observations contain measured state and RGB, never privileged state."""

from dataclasses import dataclass
from math import isfinite

from .state import EEFState, JointState


@dataclass(frozen=True)
class RGBFrame:
    """Immutable row-major uint8 RGB (HWC), with three bytes per pixel."""

    width: int
    height: int
    data: bytes

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("Image dimensions must be integers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")
        if not isinstance(self.data, bytes) or len(self.data) != self.width * self.height * 3:
            raise ValueError("Expected packed uint8 RGB bytes")


@dataclass(frozen=True)
class RobotObservation:
    """Timestamp in seconds since reset; cameras sampled at this simulation instant."""

    timestamp: float
    eef: EEFState
    head_rgb: RGBFrame
    left_wrist_rgb: RGBFrame
    right_wrist_rgb: RGBFrame
    #: Measured joint configuration, when the backend reports one. Optional so
    #: that observations reconstructed from older recordings -- which predate
    #: joint capture -- remain constructible, and so a backend without joint
    #: feedback stays usable. Consumers that need joints must say so.
    joints: JointState | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("Timestamp must be finite and nonnegative")
        if not isinstance(self.eef, EEFState):
            raise ValueError("Expected measured EEFState")
        if self.joints is not None and not isinstance(self.joints, JointState):
            raise ValueError("Expected measured JointState or None")
        if not all(
            isinstance(f, RGBFrame)
            for f in (self.head_rgb, self.left_wrist_rgb, self.right_wrist_rgb)
        ):
            raise ValueError("All three RGB cameras are required")
