"""Length-prefixed message framing shared by both sides of the eval bridge.

Standard library only, deliberately: this is the one module imported by both
the simulator environment (NumPy 1.26) and the policy environment (NumPy 2.x),
so it must not depend on either. RGB frames therefore travel as raw bytes with
their shape declared in the JSON header, and each side reconstructs them with
whatever array library it actually has.

Wire format, repeated per message:

    4 bytes   big-endian unsigned header length
    N bytes   UTF-8 JSON header, including a "blobs" list
    ...       each blob's bytes, concatenated, in header order

Framing is explicit rather than newline- or pickle-based because the payload is
binary image data (a 320x240x3 frame is 230 KB, and there are three per step):
newlines would need escaping, and pickle would let either side execute code
chosen by the other.
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass, field

#: Big-endian uint32 header length. Four bytes caps a header at 4 GiB, far
#: beyond any plausible JSON header; blobs are not subject to this limit.
_HEADER_STRUCT = struct.Struct(">I")

#: Refuse absurd headers rather than trying to allocate them. A header is
#: metadata only (op name, shapes, scalars), so anything past a megabyte means
#: a desynchronised stream, not a large message.
MAX_HEADER_BYTES = 1 << 20


class ProtocolError(RuntimeError):
    """The peer sent something unreadable, or closed mid-message."""


@dataclass(frozen=True)
class Blob:
    """One binary payload: raw bytes plus the shape/dtype needed to rebuild it."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Blob name must be non-empty")
        if not isinstance(self.data, bytes):
            raise ValueError(f"Blob {self.name!r} data must be bytes")


@dataclass(frozen=True)
class Message:
    """A decoded message: its JSON header fields and any binary blobs."""

    header: dict = field(default_factory=dict)
    blobs: tuple[Blob, ...] = ()

    @property
    def op(self) -> str:
        op = self.header.get("op")
        if not isinstance(op, str):
            raise ProtocolError(f"Message header has no string 'op': {self.header!r}")
        return op

    def blob(self, name: str) -> Blob:
        for blob in self.blobs:
            if blob.name == name:
                return blob
        raise ProtocolError(f"Message has no blob named {name!r}")


def encode(header: dict, blobs: tuple[Blob, ...] = ()) -> bytes:
    """Serialise one message. `header` must not already carry a "blobs" key."""
    if "blobs" in header:
        raise ValueError("'blobs' is reserved; it is derived from the blobs argument")
    descriptors = [
        {"name": b.name, "dtype": b.dtype, "shape": list(b.shape), "nbytes": len(b.data)}
        for b in blobs
    ]
    payload = json.dumps({**header, "blobs": descriptors}).encode("utf-8")
    if len(payload) > MAX_HEADER_BYTES:
        raise ProtocolError(f"Header of {len(payload)} bytes exceeds {MAX_HEADER_BYTES}")
    return b"".join((_HEADER_STRUCT.pack(len(payload)), payload, *(b.data for b in blobs)))


def _recv_exactly(sock: socket.socket, count: int) -> bytes:
    """Read exactly `count` bytes, or fail. recv() may return short reads."""
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ProtocolError(f"Peer closed after {count - remaining} of {count} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send(sock: socket.socket, header: dict, blobs: tuple[Blob, ...] = ()) -> None:
    sock.sendall(encode(header, blobs))


def recv(sock: socket.socket) -> Message:
    """Read one whole message, blocking until it has arrived."""
    (length,) = _HEADER_STRUCT.unpack(_recv_exactly(sock, _HEADER_STRUCT.size))
    if length > MAX_HEADER_BYTES:
        raise ProtocolError(f"Peer announced a {length}-byte header; stream is desynchronised")
    try:
        header = json.loads(_recv_exactly(sock, length).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"Header is not valid JSON: {error}") from error
    if not isinstance(header, dict):
        raise ProtocolError(f"Header must be a JSON object, got {type(header).__name__}")

    descriptors = header.pop("blobs", [])
    if not isinstance(descriptors, list):
        raise ProtocolError("Header 'blobs' must be a list")
    blobs = []
    for descriptor in descriptors:
        try:
            name, dtype = descriptor["name"], descriptor["dtype"]
            shape, nbytes = tuple(descriptor["shape"]), descriptor["nbytes"]
        except (TypeError, KeyError) as error:
            raise ProtocolError(f"Malformed blob descriptor {descriptor!r}") from error
        blobs.append(Blob(name, dtype, shape, _recv_exactly(sock, nbytes)))
    return Message(header, tuple(blobs))


def send_error(sock: socket.socket, error: BaseException) -> None:
    """Report a server-side failure to the client instead of dropping the
    connection, so the client raises something describing the real cause
    rather than a bare truncated-stream error."""
    send(sock, {"op": "error", "type": type(error).__name__, "message": str(error)})


def raise_for_error(message: Message) -> Message:
    """Re-raise a peer-reported error locally; pass any other message through."""
    if message.header.get("op") == "error":
        raise ProtocolError(
            f"Simulator raised {message.header.get('type')}: {message.header.get('message')}"
        )
    return message
