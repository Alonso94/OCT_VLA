"""The recorder runs inside evaluation, so a fault in it would abort a rollout
that costs half an hour of simulator time. These pin the parts that can fail
without a GPU or a simulator."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from oct_vla.serve.video import DEFAULT_LAYOUT, Caption, RolloutVideo, tile

np = pytest.importorskip("numpy")


def blob(height=8, width=10, value=7):
    data = bytes([value]) * (height * width * 3)
    return NS(data=data, height=height, width=width)


def observation(height=8, width=10):
    frames = {name: blob(height, width, i + 1) for i, name in enumerate(DEFAULT_LAYOUT)}
    return NS(frame=frames.__getitem__)


def test_the_scene_view_comes_first():
    """Left-to-right order is what the caption implicitly labels, so it is
    fixed rather than incidental."""
    assert DEFAULT_LAYOUT[0] == "head_camera"
    assert DEFAULT_LAYOUT[1:] == ("left_wrist_camera", "right_wrist_camera")


def test_tiling_places_panels_side_by_side():
    panels = [np.full((8, 10, 3), v, dtype=np.uint8) for v in (1, 2, 3)]
    out = tile(panels)
    assert out.shape == (8, 30, 3)
    assert out[0, 0, 0] == 1 and out[0, 10, 0] == 2 and out[0, 20, 0] == 3


def test_mismatched_heights_are_refused_not_padded():
    """Padding would leave the panels no longer lined up with their labels."""
    panels = [np.zeros((8, 10, 3), np.uint8), np.zeros((6, 10, 3), np.uint8)]
    with pytest.raises(ValueError, match="differing heights"):
        tile(panels)


def test_closing_without_frames_writes_no_file(tmp_path):
    """An episode that dies before its first step should leave nothing to
    explain, rather than a zero-byte mp4."""
    video = RolloutVideo(tmp_path / "empty.mp4")
    assert video.close() is None
    assert not (tmp_path / "empty.mp4").exists()


def test_a_rollout_is_written_and_is_playable(tmp_path):
    pytest.importorskip("av")
    path = tmp_path / "cell.mp4"
    with RolloutVideo(path, fps=15.0, caption=Caption("act rgb", "absolute joint")) as video:
        for step in range(5):
            video.add(observation(), {"step": step, "transfers": 0})
        assert video.frames == 5
    assert path.exists() and path.stat().st_size > 0

    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Odd tile widths are padded to even, which h264's yuv420p requires.
        assert stream.codec_context.width % 2 == 0
        assert stream.codec_context.height % 2 == 0
        assert sum(1 for _ in container.decode(video=0)) == 5


def test_an_odd_width_is_padded_rather_than_rejected(tmp_path):
    pytest.importorskip("av")
    path = tmp_path / "odd.mp4"
    with RolloutVideo(path) as video:
        video.add(observation(height=7, width=5))
    import av

    with av.open(str(path)) as container:
        ctx = container.streams.video[0].codec_context
        assert (ctx.width, ctx.height) == (16, 8)  # 3*5=15 -> 16, 7 -> 8
