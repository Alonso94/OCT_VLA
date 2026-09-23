"""Policy-side client: drives a remote shelf-restock simulator.

Imports nothing from RoboTwin, SAPIEN or cuRobo -- that is the whole point of
the bridge, and the reason this module can live in the LeRobot environment. It
speaks only the core dataclasses and raw RGB bytes, leaving the caller to turn
frames into whatever tensor type its policy wants.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from dataclasses import dataclass, field

from oct_vla.core.objects import ObjectScene, TaskContext
from oct_vla.core.observation import RGBFrame
from oct_vla.core.state import EEFState, JointState
from oct_vla.serve import protocol
from oct_vla.serve.codec import (
    context_from_json,
    eef_from_json,
    joints_from_json,
    scene_from_json,
    supports_from_json,
)

#: Camera order is fixed by the wire format and matches the canonical episode
#: schema, so a policy's image inputs line up with what it was trained on.
CAMERA_NAMES = ("head_camera", "left_wrist_camera", "right_wrist_camera")


@dataclass(frozen=True)
class RemoteObservation:
    """One simulator observation, as the policy environment sees it."""

    timestamp: float
    eef: EEFState
    #: Measured joint configuration, when the simulator reports one. What a
    #: joint-space policy conditions on; None for an older server.
    joints: JointState | None
    cameras: dict[str, RGBFrame]
    scene: ObjectScene
    context: TaskContext
    supports: tuple = ()
    #: track_id -> spawned mesh variant. Reported on reset only; empty after a
    #: step, and empty from a server too old to report it.
    model_ids: dict[str, int] = field(default_factory=dict)

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
    #: Free-text elaboration on `reason`, when the server supplies one -- for an
    #: unreachable pose, which arm and which pose. Carried through because the
    #: first sweep recorded 1064 unreachable_pose episodes and threw away the
    #: one field that said which arm was at fault.
    detail: str = ""
    #: Steps where a command was kinematically infeasible and the arm held.
    infeasible_steps: int = 0
    #: Objects ever raised clear of the lower shelf -- the coarsest credit.
    objects_lifted: int = 0
    #: Objects in the scene, so the three scores can be read as fractions.
    objects_total: int = 0


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
        joints=joints_from_json(header.get("joints")),
        cameras=cameras,
        scene=scene_from_json(header["scene"]),
        context=context_from_json(header["context"]),
        supports=supports_from_json(header.get("supports", [])),
        model_ids={str(k): int(v) for k, v in (header.get("model_ids") or {}).items()},
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
        #: Set by reset(); step() validates the action width against it.
        self._control_space = "cartesian"

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

    def reset(
        self,
        seed: int,
        profile: str = "three_object",
        control_space: str = "cartesian",
        gripper_encoding: str = "measured_aperture",
        model_ids: Sequence[int] | None = None,
    ) -> RemoteObservation:
        """Start a fresh scene. Raises if the simulator cannot build that seed.

        `control_space` tells the server how to read the actions that follow:
        "cartesian" for 14-d canonical increments executed through IK, "joint"
        for absolute joint targets commanded directly. Declared rather than
        inferred from the vector's width, so a policy whose action space does
        not match the robot fails loudly instead of being misread.

        `model_ids` pins which mesh variants the scene may spawn. It is checked
        here as well as on the server, because a server that predates the field
        ignores it without complaint -- and a silently unpinned rollout is
        indistinguishable from a pinned one in every score it reports.
        """
        self._control_space = control_space
        header = {
            "op": "reset",
            "seed": seed,
            "profile": profile,
            "control_space": control_space,
            "gripper_encoding": gripper_encoding,
        }
        if model_ids is not None:
            header["model_ids"] = [int(v) for v in model_ids]
        observation = _observation(self._round_trip(header))
        if model_ids is not None:
            if not observation.model_ids:
                raise protocol.ProtocolError(
                    f"requested model_ids {list(model_ids)} but the server reported no "
                    "spawned variants; it may predate the field and have ignored the pin"
                )
            stray = {t: v for t, v in observation.model_ids.items() if v not in set(model_ids)}
            if stray:
                raise protocol.ProtocolError(
                    f"requested model_ids {list(model_ids)} but the scene holds {stray}"
                )
        return observation

    def step(self, action: Sequence[float]) -> StepResult:
        """Apply one action in whatever space `reset` declared.

        Cartesian is the 14-D canonical action (docs/canonical_action.md);
        joint is an absolute configuration, whose width follows the embodiment
        and is checked against the robot server-side.
        """
        values = [float(v) for v in action]
        if self._control_space == "cartesian" and len(values) != 14:
            raise ValueError(f"Canonical action must have 14 elements; got {len(values)}")
        if self._control_space == "cartesian_absolute" and len(values) != 16:
            raise ValueError(
                f"An absolute end-effector action must have 16 elements "
                f"(position, quaternion, gripper, per arm); got {len(values)}"
            )
        message = self._round_trip({"op": "step", "action": values})
        return StepResult(
            observation=_observation(message),
            success=bool(message.header["success"]),
            done=bool(message.header["done"]),
            reason=str(message.header.get("reason", "")),
            transfers_completed=int(message.header.get("transfers_completed", 0)),
            detail=str(message.header.get("detail", "")),
            infeasible_steps=int(message.header.get("infeasible_steps", 0)),
            objects_lifted=int(message.header.get("objects_lifted", 0)),
            objects_total=int(message.header.get("objects_total", 0)),
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
