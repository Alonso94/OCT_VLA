"""Sharing inodes between datasets is safe only if it is driven by content.
Linking on a name or a size match would silently give two variants the same
camera stream when they do not have one."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "dedupe", ROOT / "scripts/dedupe_dataset_videos.py"
)
dedupe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dedupe)


def dataset(root: Path, name: str, payload: bytes, camera: str = "head") -> Path:
    d = root / name / "videos" / f"observation.images.{camera}" / "chunk-000"
    d.mkdir(parents=True)
    (d / "file-000.mp4").write_bytes(payload)
    return root / name


def test_identical_files_are_shared(tmp_path):
    a = dataset(tmp_path, "rgb", b"same-bytes" * 1000)
    b = dataset(tmp_path, "rgb_abs", b"same-bytes" * 1000)
    groups = dedupe.plan([a, b])
    (key, paths), = [(k, v) for k, v in groups.items() if len(v) > 1]
    assert len(paths) == 2
    reclaimed = dedupe.link(paths[0], paths[1])
    assert reclaimed == 10000
    assert paths[0].stat().st_ino == paths[1].stat().st_ino
    assert paths[1].read_bytes() == b"same-bytes" * 1000


def test_differing_content_is_never_shared(tmp_path):
    """Same relative path, same size, different bytes -- must stay separate."""
    a = dataset(tmp_path, "rgb", b"A" * 500)
    b = dataset(tmp_path, "rgb_abs", b"B" * 500)
    assert all(len(v) == 1 for v in dedupe.plan([a, b]).values())


def test_a_different_camera_is_not_shared_with_another(tmp_path):
    """Keyed on the relative path too, so two cameras that happened to encode
    identically still keep their own file and the tree stays self-explanatory."""
    a = dataset(tmp_path, "rgb", b"x" * 100, camera="head")
    b = dataset(tmp_path, "rgb", b"x" * 100, camera="left_wrist")
    assert a == b
    assert all(len(v) == 1 for v in dedupe.plan([a]).values())


def test_relinking_an_already_shared_file_is_a_no_op(tmp_path):
    a = dataset(tmp_path, "rgb", b"z" * 400)
    b = dataset(tmp_path, "rgb_abs", b"z" * 400)
    first = a / "videos/observation.images.head/chunk-000/file-000.mp4"
    second = b / "videos/observation.images.head/chunk-000/file-000.mp4"
    dedupe.link(first, second)
    assert dedupe.link(first, second) == 0


def test_the_link_leaves_no_temporary_behind(tmp_path):
    a = dataset(tmp_path, "rgb", b"q" * 300)
    b = dataset(tmp_path, "rgb_abs", b"q" * 300)
    first = a / "videos/observation.images.head/chunk-000/file-000.mp4"
    second = b / "videos/observation.images.head/chunk-000/file-000.mp4"
    dedupe.link(first, second)
    assert [p.name for p in second.parent.iterdir()] == ["file-000.mp4"]
    assert not any(str(p).endswith(".linking") for p in tmp_path.rglob("*"))


def test_content_hash_does_not_depend_on_read_chunking(tmp_path):
    big = dataset(tmp_path, "rgb", os.urandom(3 * dedupe.CHUNK + 17))
    path = big / "videos/observation.images.head/chunk-000/file-000.mp4"
    import hashlib
    assert dedupe.digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()
