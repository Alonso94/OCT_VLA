import json
import socket
import struct
import threading

import pytest

from oct_vla.serve import protocol


def _pair() -> tuple[socket.socket, socket.socket]:
    """A connected socket pair, so framing is exercised over a real stream
    rather than a buffer that cannot produce short reads."""
    return socket.socketpair()


def test_round_trip_header_only():
    left, right = _pair()
    with left, right:
        protocol.send(left, {"op": "step", "action": [0.0] * 14})
        message = protocol.recv(right)
    assert message.op == "step"
    assert message.header["action"] == [0.0] * 14
    assert message.blobs == ()


def test_round_trip_with_blobs():
    left, right = _pair()
    head = bytes(range(256)) * 4
    wrist = bytes(reversed(range(256))) * 4
    with left, right:
        protocol.send(
            left,
            {"op": "obs"},
            (
                protocol.Blob("head_camera", "uint8", (2, 2, 3), head),
                protocol.Blob("left_wrist_camera", "uint8", (2, 2, 3), wrist),
            ),
        )
        message = protocol.recv(right)
    # Distinct payloads: a concatenation-offset bug would swap or splice them
    # rather than produce equal-but-wrong bytes.
    assert message.blob("head_camera").data == head
    assert message.blob("left_wrist_camera").data == wrist
    assert message.blob("head_camera").shape == (2, 2, 3)


def test_large_payload_survives_short_reads():
    """A 320x240x3 frame exceeds the socket buffer, so recv() returns partial
    chunks; _recv_exactly must reassemble them."""
    payload = bytes(320 * 240 * 3)
    left, right = _pair()
    with left, right:
        sender = threading.Thread(
            target=protocol.send,
            args=(
                left,
                {"op": "obs"},
                (protocol.Blob("head_camera", "uint8", (240, 320, 3), payload),),
            ),
        )
        sender.start()
        message = protocol.recv(right)
        sender.join()
    assert len(message.blob("head_camera").data) == len(payload)


def test_blobs_key_is_reserved():
    with pytest.raises(ValueError, match="reserved"):
        protocol.encode({"op": "obs", "blobs": []})


def test_missing_blob_is_an_error():
    message = protocol.Message({"op": "obs"}, ())
    with pytest.raises(protocol.ProtocolError, match="no blob named"):
        message.blob("head_camera")


def test_header_without_op_is_rejected():
    with pytest.raises(protocol.ProtocolError, match="no string 'op'"):
        _ = protocol.Message({"seed": 1}, ()).op


def test_peer_close_mid_message_raises():
    left, right = _pair()
    with left, right:
        left.sendall(struct.Struct(">I").pack(64))  # promise 64 bytes, send none
        left.close()
        with pytest.raises(protocol.ProtocolError, match="closed after"):
            protocol.recv(right)


def test_absurd_header_length_is_refused_without_allocating():
    left, right = _pair()
    with left, right:
        left.sendall(struct.Struct(">I").pack(protocol.MAX_HEADER_BYTES + 1))
        with pytest.raises(protocol.ProtocolError, match="desynchronised"):
            protocol.recv(right)


def test_non_json_header_raises_protocol_error():
    left, right = _pair()
    with left, right:
        body = b"not json"
        left.sendall(struct.Struct(">I").pack(len(body)) + body)
        with pytest.raises(protocol.ProtocolError, match="not valid JSON"):
            protocol.recv(right)


def test_json_header_must_be_an_object():
    left, right = _pair()
    with left, right:
        body = json.dumps([1, 2, 3]).encode()
        left.sendall(struct.Struct(">I").pack(len(body)) + body)
        with pytest.raises(protocol.ProtocolError, match="must be a JSON object"):
            protocol.recv(right)


def test_error_messages_reraise_locally():
    left, right = _pair()
    with left, right:
        protocol.send_error(left, ValueError("seed 42 is unreachable"))
        message = protocol.recv(right)
        with pytest.raises(protocol.ProtocolError, match="ValueError.*seed 42 is unreachable"):
            protocol.raise_for_error(message)


def test_raise_for_error_passes_normal_messages_through():
    message = protocol.Message({"op": "step"}, ())
    assert protocol.raise_for_error(message) is message


def test_two_messages_back_to_back_stay_framed():
    """Streams are not message-oriented: the second message must not be read as
    a continuation of the first."""
    left, right = _pair()
    with left, right:
        protocol.send(left, {"op": "a"}, (protocol.Blob("x", "uint8", (1,), b"\x01"),))
        protocol.send(left, {"op": "b"})
        first, second = protocol.recv(right), protocol.recv(right)
    assert (first.op, second.op) == ("a", "b")
    assert first.blob("x").data == b"\x01"
