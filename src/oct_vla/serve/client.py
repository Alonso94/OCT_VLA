"""Policy-side client: drives a remote shelf-restock simulator.

Imports nothing from RoboTwin, SAPIEN or cuRobo -- that is the whole point of
the bridge, and the reason this module can live in the LeRobot environment. It
speaks only the core dataclasses and raw RGB bytes, leaving the caller to turn
frames into whatever tensor type its policy wants.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from dataclasses import dataclass

from oct_vla.core.objects import ObjectScene, TaskContext
from oct_vla.core.observation import RGBFrame
from oct_vla.core.state import EEFState
from oct_vla.serve import protocol
from oct_vla.serve.codec import context_from_json, eef_from_json, scene_from_json

#: Camera order is fixed by the wire format and matches the canonical episode
#: schema, so a policy's image inputs line up with what it was trained on.
CAMERA_NAMES = ("head_camera", "left_wrist_camera", "right_wrist_camera")


@dataclass(frozen=True)
class RemoteObservation:
    """One simulator observation, as the policy environment sees it."""

    timestamp: float
    eef: EEFState
    cameras: dict[str, RGBFrame]
    scene: ObjectScene
    context: TaskContext

    def frame(self, name: str) -> RGBFrame:
        try:
            return self.cameras[name]
        except KeyError:
            raise KeyError(f"No camera {name!r}; have {sorted(self.cameras)}") from None


@dataclass(frozen=True)
class StepResult:
    observation: RemoteObservation
    success: bool
    done: bool
    #: Why the episode ended: "success", "step_limit", or a failure reason.
    reason: str
    #: Transfers completed so far, for partial-credit reporting.
    transfers_completed: int


def _observation(message: protocol.Message) -> RemoteObservation:
    header = message.header
    cameras = {}
    for name in CAMERA_NAMES:
        blob = message.blob(name)
        height, width, channels = blob.shape
        if channels != 3 or blob.dtype != "uint8":
            raise protocol.ProtocolError(
                f"{name} must be uint8 HWC with 3 channels; got {blob.dtype} {blob.shape}"
            )
        cameras[name] = RGBFrame(width, height, blob.data)
    return RemoteObservation(
        timestamp=header["timestamp"],
        eef=eef_from_json(header["eef"]),
        cameras=cameras,
        scene=scene_from_json(header["scene"]),
        context=context_from_json(header["context"]),
    )


class ShelfRestockEvalClient:
    """Connects to a running simulator server and runs closed-loop episodes.

    Use as a context manager so the socket is closed even when a rollout raises;
    a leaked connection leaves the server waiting on a peer that will never
    speak again, and the next evaluation would block on connect.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, *, timeout: float = 600.0):
        self._address = (host, port)
        self._timeout = timeout
        self._sock: socket.socket | None = None

    def __enter__(self) -> ShelfRestockEvalClient:
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def connect(self) -> None:
        sock = socket.create_connection(self._address, timeout=self._timeout)
        # Control loops send small messages and wait for the reply, which is
        # exactly the pattern Nagle's algorithm delays.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    @property
    def _connection(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError(
                "Client is not connected; call connect() or use as a context manager"
            )
        return self._sock

    def _round_trip(self, header: dict) -> protocol.Message:
        protocol.send(self._connection, header)
        return protocol.raise_for_error(protocol.recv(self._connection))

    def reset(self, seed: int, profile: str = "three_object") -> RemoteObservation:
        """Start a fresh scene. Raises if the simulator cannot build that seed."""
        return _observation(self._round_trip({"op": "reset", "seed": seed, "profile": profile}))

    def step(self, action: Sequence[float]) -> StepResult:
        """Apply one 14-D canonical action (see docs/canonical_action.md)."""
        values = [float(v) for v in action]
        if len(values) != 14:
            raise ValueError(f"Canonical action must have 14 elements; got {len(values)}")
        message = self._round_trip({"op": "step", "action": values})
        return StepResult(
            observation=_observation(message),
            success=bool(message.header["success"]),
            done=bool(message.header["done"]),
            reason=str(message.header.get("reason", "")),
            transfers_completed=int(message.header.get("transfers_completed", 0)),
        )

    def close(self) -> None:
        """Tell the server this client is done, then hang up. Never raises: it
        runs on the error path too, where the connection may already be gone."""
        if self._sock is None:
            return
        try:
            protocol.send(self._sock, {"op": "close"})
        except OSError:
            pass
        finally:
            self._sock.close()
            self._sock = None
