"""The identity tier must reach the scene, and a rollout must prove it did.

Regression for the final matrix: the job exported the pin after forking the
simulator server, so the server never saw it, and all 44 seen/held-out/novel
rollouts scored the same mixed-identity scenes -- identical episode by episode,
with nothing in any result file to show it.
"""

from __future__ import annotations

import os
import socket
import threading

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, TaskContext
from oct_vla.core.state import ArmState, EEFState
from oct_vla.serve import protocol
from oct_vla.serve.client import ShelfRestockEvalClient
from oct_vla.serve.codec import context_to_json, eef_to_json, scene_to_json
from oct_vla.serve.server import (
    MODEL_IDS_ENV,
    EvalServerError,
    ShelfRestockEvalServer,
    _pinned_model_ids,
    _spawned_model_ids,
)


def _observation_header() -> dict:
    pose = Pose((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME)
    return {
        "timestamp": 0.0,
        "eef": eef_to_json(EEFState(ArmState(pose, 0.0), ArmState(pose, 0.0))),
        "scene": scene_to_json(ObjectScene(0.0, ())),
        "context": context_to_json(TaskContext("Restock.", "obj_0", None, "approach")),
    }


def _frames() -> tuple[protocol.Blob, ...]:
    return tuple(
        protocol.Blob(name, "uint8", (1, 1, 3), b"\x00\x00\x00")
        for name in ("head_camera", "left_wrist_camera", "right_wrist_camera")
    )


class Task:
    def __init__(self, spawned):
        self.spawned_model_ids = spawned
        self.tracked_objects = {}


class PinReadingPort:
    """Spawns whatever the environment pin allows, as `_load_objects` does."""

    def __init__(self, ignore_pin: bool = False):
        self.ignore_pin = ignore_pin
        self.seen_pins: list[str | None] = []
        self.task = None

    def reset(self, seed):
        pin = os.environ.get(MODEL_IDS_ENV)
        self.seen_pins.append(pin)
        allowed = [int(v) for v in pin.split(",")] if pin and not self.ignore_pin else [0, 5]
        self.task = Task({f"obj_{i}": allowed[i % len(allowed)] for i in range(3)})

    def close(self):
        pass


def server_on(port) -> ShelfRestockEvalServer:
    server = ShelfRestockEvalServer(lambda task: port)
    server._snapshot = lambda: ({}, (), None)
    return server


def test_the_pin_is_visible_to_the_scene_build_and_only_to_it(monkeypatch):
    monkeypatch.delenv(MODEL_IDS_ENV, raising=False)
    port = PinReadingPort()
    header, _ = server_on(port).reset(800, "three_object", 600, model_ids=(0, 6))
    assert port.seen_pins == ["0,6"]
    assert set(header["model_ids"].values()) <= {0, 6}
    assert MODEL_IDS_ENV not in os.environ, "the pin must not leak into the next reset"


def test_an_unpinned_reset_clears_an_inherited_pin(monkeypatch):
    monkeypatch.setenv(MODEL_IDS_ENV, "5")
    port = PinReadingPort()
    server_on(port).reset(800, "three_object", 600)
    assert port.seen_pins == [None]
    assert os.environ[MODEL_IDS_ENV] == "5", "the caller's environment is restored"


def test_a_scene_outside_the_pin_is_refused():
    port = PinReadingPort(ignore_pin=True)
    with pytest.raises(EvalServerError, match="spawned"):
        server_on(port).reset(800, "three_object", 600, model_ids=(1, 2, 3, 4))


def test_a_task_that_cannot_report_its_variants_is_refused():
    with pytest.raises(EvalServerError, match="cannot be verified"):
        _spawned_model_ids(Task({}), (1, 2))
    assert _spawned_model_ids(Task({}), None) == {}


def test_the_pin_context_restores_on_error(monkeypatch):
    monkeypatch.delenv(MODEL_IDS_ENV, raising=False)
    with pytest.raises(RuntimeError), _pinned_model_ids((3,)):
        assert os.environ[MODEL_IDS_ENV] == "3"
        raise RuntimeError
    assert MODEL_IDS_ENV not in os.environ


def reset_against(header: dict, **kwargs):
    server_sock, client_sock = socket.socketpair()
    client = ShelfRestockEvalClient()
    client._sock = client_sock
    requests = []

    def serve():
        requests.append(protocol.recv(server_sock).header)
        protocol.send(server_sock, header, _frames())

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        return client.reset(seed=800, profile="three_object", **kwargs), requests
    finally:
        thread.join()
        client_sock.close()
        server_sock.close()


def test_the_client_sends_the_pin_and_returns_what_spawned():
    header = {**_observation_header(), "model_ids": {"obj_0": 0, "obj_1": 6}}
    observation, requests = reset_against(header, model_ids=[0, 6])
    assert requests[0]["model_ids"] == [0, 6]
    assert observation.model_ids == {"obj_0": 0, "obj_1": 6}


def test_the_client_refuses_a_server_that_ignored_the_pin():
    """The exact failure: an older server drops the unknown field silently."""
    with pytest.raises(protocol.ProtocolError, match="ignored the pin"):
        reset_against(_observation_header(), model_ids=[5])


def test_the_client_refuses_a_scene_outside_the_pin():
    header = {**_observation_header(), "model_ids": {"obj_0": 1}}
    with pytest.raises(protocol.ProtocolError, match="scene holds"):
        reset_against(header, model_ids=[5])


def test_an_unpinned_reset_sends_no_pin():
    _, requests = reset_against(_observation_header())
    assert "model_ids" not in requests[0]
