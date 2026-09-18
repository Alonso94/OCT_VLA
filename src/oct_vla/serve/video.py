"""Record a closed-loop rollout as an mp4, from the frames the bridge already sends.

The evaluation harness reports a rollout as counters -- transfers, lifts, the
reason it stopped. Those separate "did nothing" from "did the task" but not
"reached and missed" from "never approached", and with several action encodings
and observation variants in flight the difference between them is the whole
question. A video answers it in seconds.

The frames come free: the server sends all three cameras on every step whatever
the policy reads, so a vision-free privileged checkpoint records exactly as well
as an RGB one.

Three views side by side rather than the scene view alone. The head camera shows
where the arm went; the wrist cameras show whether the gripper closed on
anything, which is the distinction the counters are worst at and the one the
lift/transfer gap turns on.

PyAV rather than imageio: LeRobot already depends on it, so it is present in the
policy environment wherever evaluation runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Bridge camera names, left to right in the tile. Scene view first.
DEFAULT_LAYOUT = ("head_camera", "left_wrist_camera", "right_wrist_camera")


@dataclass(frozen=True)
class Caption:
    """The fixed banner identifying which cell a video belongs to."""

    title: str
    subtitle: str = ""


def tile(frames: list) -> object:
    """Lay decoded frames out in one row.

    Requires equal heights, which the three cameras have by construction. A
    mismatch means the layout named a camera the server renders differently, and
    silently padding it would produce a video whose panels no longer line up
    with their labels.
    """
    import numpy as np

    heights = {frame.shape[0] for frame in frames}
    if len(heights) != 1:
        raise ValueError(f"Cannot tile frames of differing heights: {sorted(heights)}")
    return np.concatenate(frames, axis=1)


class RolloutVideo:
    """Writes one mp4 for one episode, one frame per control step.

    Opened lazily on the first frame so the encoder is configured from the real
    frame size rather than an assumption, and so an episode that dies before its
    first step leaves no empty file to explain.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float = 15.0,
        layout: tuple[str, ...] = DEFAULT_LAYOUT,
        caption: Caption | None = None,
        crf: int = 30,
    ):
        self.path = Path(path)
        self.fps = fps
        self.layout = layout
        self.caption = caption
        self.crf = crf
        self._container = None
        self._stream = None
        self.frames = 0

    # ------------------------------------------------------------------ setup
    def _open(self, width: int, height: int) -> None:
        import av

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.path), mode="w")
        # h264 with yuv420p is what every browser and every viewer in this
        # repository's docs can play without a plugin. Both dimensions must be
        # even for that pixel format, so an odd tile is padded by one column
        # rather than rejected.
        self._stream = self._container.add_stream("libx264", rate=int(round(self.fps)))
        self._stream.width = width + (width % 2)
        self._stream.height = height + (height % 2)
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": str(self.crf), "preset": "veryfast"}

    # ------------------------------------------------------------------ write
    def add(self, observation, overlay: dict | None = None) -> None:
        """Decode this step's cameras, tile them, annotate and encode one frame."""
        import av
        import numpy as np

        panels = []
        for camera in self.layout:
            blob = observation.frame(camera)
            panels.append(
                np.frombuffer(blob.data, dtype=np.uint8)
                .reshape(blob.height, blob.width, 3)
                .copy()
            )
        image = tile(panels)
        image = self._annotate(image, overlay or {})

        if self._container is None:
            self._open(image.shape[1], image.shape[0])
        if image.shape[1] != self._stream.width or image.shape[0] != self._stream.height:
            padded = np.zeros((self._stream.height, self._stream.width, 3), dtype=np.uint8)
            padded[: image.shape[0], : image.shape[1]] = image
            image = padded

        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames += 1

    def _annotate(self, image, overlay: dict):
        """Burn the caption and live counters into the frame.

        Burned in rather than written to a sidecar: these files are meant to be
        opened one after another out of a directory listing, and a video that
        cannot say which cell it belongs to is worse than useless in a grid of
        seventeen.

        Missing OpenCV is not fatal -- an unlabelled video still shows the
        behaviour, and failing the whole rollout over a text overlay would be
        the wrong trade.
        """
        try:
            import cv2
        except ModuleNotFoundError:
            return image

        lines = []
        if self.caption is not None:
            lines.append((self.caption.title, 0.5, (255, 255, 255)))
            if self.caption.subtitle:
                lines.append((self.caption.subtitle, 0.4, (185, 205, 235)))
        if overlay:
            lines.append((
                "  ".join(f"{key} {value}" for key, value in overlay.items()),
                0.4,
                (170, 235, 170),
            ))

        y = 14
        for text, scale, colour in lines:
            # Black underlay first, so the text stays readable over both the
            # bright shelf and the dark background.
            cv2.putText(image, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(image, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1,
                        cv2.LINE_AA)
            y += 15
        return image

    # ------------------------------------------------------------------ close
    def close(self) -> Path | None:
        """Flush the encoder. Returns the path written, or None if no frames were."""
        if self._container is None:
            return None
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = None
        self._stream = None
        return self.path

    def __enter__(self) -> RolloutVideo:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
