"""Recorded RoboTwin episodes -> the same physical ``Episode`` the generator produces.

Reads ``episode.json`` only (no camera decoding). Columns:
- both wrists: the recorded EEF pose (xyzw), the convention every pose-window table uses;
- movables: every tracked object, with colour (``ObjectState.colour`` when the recording
  has it, else unknown) and category mapped from the asset name;
- fixtures: the shelf decks (``SHELF_DECKS``) for shelf runs.

Time is the recorded timestamp of each frame, not ``index * DT``: frames are 16 or 17
physics ticks apart, so the measured interval is used everywhere.

**Target provenance.** A legacy recording has no commanded wrist stream, so the target at
row t+h is the *measured* wrist at that row (``target_source = "next_measured_pose"``);
it is exactly the two pose blocks of ``lerobot_export._state_vector(ep, t + h)``.
Recordings with ``episode.commands.npz`` (plan 0.1) add the commanded stream separately.

The recorder's ``phase`` and ``context`` labels are oracle information: they go to the
evaluator-only annotations, never to a table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oct_vla.bpfn import quat
from oct_vla.bpfn.schema import FIXTURE, MOVABLE, Episode, category_id, colour_id

#: Asset name -> stable category. Unlisted assets are unknown, never guessed.
CATEGORY_OF_ASSET = {
    "113_coffee-box": "box",
    "071_can": "can",
    "105_sauce-can": "can",
    "018_microphone": "microphone",
    "003_plate": "plate",
    "062_plasticbox": "container",
    "002_bowl": "container",
    "021_cup": "container",
}
TARGET_SOURCE_LEGACY = "next_measured_pose"


#: The shelf decks as fixtures: (name, centre xyz, size xyz), identity orientation. They
#: equal ``tasks.shelf_restock.spec.DEFAULT_SPEC`` (tests/bpfn/test_standalone.py holds
#: the two together).
SHELF_DECKS = (("lower_deck", (-0.07, -0.25, 0.74), (1.0, 0.2, 0.02)),
               ("upper_deck", (-0.06, -0.06, 0.92), (0.6, 0.12, 0.03)))


def _pose7(pose: dict) -> np.ndarray:
    return np.array([*pose["position"], *pose["orientation"]], float)


def recorded_metadata(directory: str | Path) -> dict:
    """The run's metadata (task, rule, profile, ...) from ``episode.json``."""
    payload = json.loads((Path(directory) / "episode.json").read_text())
    return dict(payload["metadata"])


def episode_from_json(payload: dict, shelf: bool = True) -> Episode:
    """A recorded run's ``episode.json`` payload as a physical ``Episode`` (states only;
    the camera files are never opened). ``shelf``: add the decks as fixtures."""
    samples = payload["samples"]
    T = len(samples)
    first = samples[0]["scene"]["objects"]
    ids = tuple(o["track_id"] for o in first)

    wrist = np.zeros((T, 2, 7))
    aperture = np.zeros((T, 2))
    objects = np.zeros((T, len(ids), 7))
    for t, sample in enumerate(samples):
        for a, side in enumerate(("left", "right")):
            wrist[t, a] = _pose7(sample["eef"][side]["pose"])
            aperture[t, a] = sample["eef"][side]["gripper"]
        by_id = {o["track_id"]: o for o in sample["scene"]["objects"]}
        for j, track in enumerate(ids):
            if track not in by_id:
                raise ValueError(f"seed {payload['seed']}: {track} leaves the scene at frame {t}")
            objects[t, j] = _pose7(by_id[track]["pose"])

    kind = [MOVABLE] * len(ids)
    colour = [colour_id(o.get("colour")) for o in first]
    category = [category_id(CATEGORY_OF_ASSET.get(o.get("category"))) for o in first]
    size = [tuple(o["size_xyz"]) for o in first]
    tracks = list(ids)
    if shelf:
        for name, centre, extent in SHELF_DECKS:
            deck = np.array([*centre, 0.0, 0.0, 0.0, 1.0])
            objects = np.concatenate([objects, np.tile(deck, (T, 1, 1))], axis=1)
            kind.append(FIXTURE)
            colour.append(colour_id(None))
            category.append(category_id(None))
            size.append(extent)
            tracks.append(name)

    for a in range(2):
        wrist[:, a, 3:] = quat.continuous(quat.normalize(wrist[:, a, 3:]))
    for j in range(objects.shape[1]):
        objects[:, j, 3:] = quat.continuous(quat.normalize(objects[:, j, 3:]))
    metadata = payload["metadata"]
    annotations = {
        "source": "recording", "seed": payload["seed"], "success": payload["success"],
        "rule": metadata.get("rule"), "task": metadata.get("task"),
        "target_source": TARGET_SOURCE_LEGACY,
        "phases": tuple(s["phase"] for s in samples),
        "targets": tuple(s["context"].get("target_track_id") for s in samples),
        "physical": len(ids),
    }
    times = np.array([float(s["timestamp"]) for s in samples])
    return Episode(times, wrist, aperture, objects, np.asarray(kind, np.int8),
                   np.asarray(colour, np.int8), np.asarray(category, np.int8),
                   np.asarray(size, float), tuple(tracks), annotations)


def read_recording(directory: str | Path, *, shelf: bool = True) -> Episode:
    """One recorded run as a physical ``Episode``; shelf runs get the decks as fixtures."""
    payload = json.loads((Path(directory) / "episode.json").read_text())
    is_shelf = shelf and payload["metadata"].get("task", "").lower().startswith("shelf")
    return episode_from_json(payload, shelf=is_shelf)
