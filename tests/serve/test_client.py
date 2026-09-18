import socket
import threading

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.core.state import ArmState, EEFState
from oct_vla.serve import protocol
from oct_vla.serve.client import ShelfRestockEvalClient
from oct_vla.serve.codec import (
    context_from_json,
    context_to_json,
    eef_from_json,
    eef_to_json,
    scene_from_json,
    scene_to_json,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)
WIDTH, HEIGHT = 4, 3


def _eef() -> EEFState:
    return EEFState(
        ArmState(Pose((0.1, 0.2, 0.3), IDENTITY, WORKCELL_FRAME), 0.25),
        ArmState(Pose((0.4, 0.5, 0.6), IDENTITY, WORKCELL_FRAME), 0.75),
    )


def _scene() -> ObjectScene:
    return ObjectScene(
        1.5,
        (
            ObjectState(
                track_id="obj_0",
                pose=Pose((0.1, 0.0, 0.9), IDENTITY, WORKCELL_FRAME),
                size_xyz=(0.06, 0.04, 0.12),
                visibility=1.0,
                confidence=0.9,
                support_surface="lower_shelf",
            ),
        ),
    )


def _context() -> TaskContext:
    return TaskContext(
        instruction="Restock the selected object.",
        target_track_id="obj_0",
        previous_neighbor_track_id=None,
        phase="approach",
    )


def _frames() -> tuple[protocol.Blob, ...]:
    # A distinct constant per camera, so a mis-ordered blob list is visible.
    return tuple(
        protocol.Blob(name, "uint8", (HEIGHT, WIDTH, 3), bytes([fill]) * (WIDTH * HEIGHT * 3))
        for fill, name in enumerate(("head_camera", "left_wrist_camera", "right_wrist_camera"), 1)
    )


def _observation_header() -> dict:
    return {
        "timestamp": 1.5,
        "eef": eef_to_json(_eef()),
        "scene": scene_to_json(_scene()),
        "context": context_to_json(_context()),
    }


def _serve(
    sock: socket.socket, replies: list[tuple[dict, tuple[protocol.Blob, ...]]]
) -> list[dict]:
    """Answer each request with the next canned reply; return what was asked."""
    seen = []
    for header, blobs in replies:
        message = protocol.recv(sock)
        seen.append(message.header)
        protocol.send(sock, header, blobs)
    return seen


@pytest.fixture
def connected():
    """A client wired to a fake server on the other end of a socket pair."""
    server_sock, client_sock = socket.socketpair()
    client = ShelfRestockEvalClient()
    client._sock = client_sock  # bypass connect(); no real listener needed
    yield client, server_sock
    client_sock.close()
    server_sock.close()


def test_reset_decodes_observation(connected):
    client, server_sock = connected
    requests: list[dict] = []
    thread = threading.Thread(
        target=lambda: requests.extend(_serve(server_sock, [(_observation_header(), _frames())]))
    )
    thread.start()
    observation = client.reset(seed=1000, profile="three_object")
    thread.join()

    # The server reads the action encodings out of this header, so the defaults
    # are part of the protocol: an older client must keep executing as it did.
    assert requests[0] == {
        "op": "reset",
        "seed": 1000,
        "profile": "three_object",
        "control_space": "cartesian",
        "gripper_encoding": "measured_aperture",
    }
    assert observation.timestamp == 1.5
    assert observation.eef.left.gripper == 0.25
    assert observation.eef.right.gripper == 0.75
    assert observation.scene.objects[0].track_id == "obj_0"
    assert observation.context.target_track_id == "obj_0"
    head = observation.frame("head_camera")
    assert (head.width, head.height) == (WIDTH, HEIGHT)
    assert head.data[0] == 1
    assert observation.frame("right_wrist_camera").data[0] == 3


def test_step_sends_action_and_reports_outcome(connected):
    client, server_sock = connected
    header = {**_observation_header(), "success": True, "done": True,
              "reason": "success", "transfers_completed": 3}
    requests: list[dict] = []
    thread = threading.Thread(
        target=lambda: requests.extend(_serve(server_sock, [(header, _frames())]))
    )
    thread.start()
    result = client.step([0.01] * 14)
    thread.join()

    assert requests[0]["op"] == "step"
    assert requests[0]["action"] == [0.01] * 14
    assert (result.success, result.done, result.reason) == (True, True, "success")
    assert result.transfers_completed == 3


def test_step_rejects_wrong_action_length(connected):
    client, _ = connected
    with pytest.raises(ValueError, match="14 elements"):
        client.step([0.0] * 13)


def test_server_error_surfaces_as_protocol_error(connected):
    client, server_sock = connected
    def reply():
        protocol.recv(server_sock)
        protocol.send_error(server_sock, RuntimeError("IK failed"))

    thread = threading.Thread(target=reply)
    thread.start()
    with pytest.raises(protocol.ProtocolError, match="RuntimeError.*IK failed"):
        client.reset(seed=1, profile="three_object")
    thread.join()


def test_unknown_camera_names_its_alternatives(connected):
    client, server_sock = connected
    reply = [(_observation_header(), _frames())]
    thread = threading.Thread(target=lambda: _serve(server_sock, reply))
    thread.start()
    observation = client.reset(seed=1, profile="three_object")
    thread.join()
    with pytest.raises(KeyError, match="head_camera"):
        observation.frame("wrist")


def test_step_before_connect_is_a_clear_error():
    with pytest.raises(RuntimeError, match="not connected"):
        ShelfRestockEvalClient().step([0.0] * 14)


def test_close_is_safe_without_a_connection():
    ShelfRestockEvalClient().close()  # must not raise on the error path


def test_codec_round_trips_scene_context_and_eef():
    assert scene_from_json(scene_to_json(_scene())) == _scene()
    assert context_from_json(context_to_json(_context())) == _context()
    assert eef_from_json(eef_to_json(_eef())) == _eef()


def test_object_tokens_build_from_a_decoded_scene():
    """The decoded scene must satisfy the token encoder, since that is the only
    reason the client carries object state at all."""
    from oct_vla.data.object_tokens import ObjectTokenSpec, object_tokens

    scene = scene_from_json(scene_to_json(_scene()))
    tokens, mask = object_tokens(scene, _context(), spec=ObjectTokenSpec())
    assert len(tokens) == 8 and len(tokens[0]) == 15
    assert mask[0] is True and mask[1] is False
    assert tokens[0][:3] == (0.1, 0.0, 0.9)
