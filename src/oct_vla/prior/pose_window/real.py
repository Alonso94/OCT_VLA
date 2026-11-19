"""Recorded RoboTwin episodes -> the same physical ``Episode`` the generator produces.

Reads states only (``data.store.read_states``: no camera decoding). Columns:
- both wrists: the recorded EEF pose (xyzw), the convention every pose-window table uses;
- movables: every tracked object, with colour (``ObjectState.colour`` when the recording
  has it, else unknown) and category mapped from the asset name;
- fixtures: the shelf decks (``entity_tokens.shelf_support_entities``) for shelf runs.

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

from pathlib import Path

import numpy as np

from . import quat as Q
from .schema import FIXTURE, MOVABLE, Episode, category_id, colour_id

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


def _pose7(pose) -> np.ndarray:
    return np.array([*pose.position, *pose.orientation], float)


def episode_from_states(states, *, shelf_spec=None) -> Episode:
    """``states``: a ``data.store.EpisodeStates``. ``shelf_spec``: add the shelf decks as
    fixtures (a ``ShelfRestockSpec``); None for tasks without them."""
    T = len(states)
    ids = tuple(o.track_id for o in states.scenes[0].objects)
    wrist = np.array([[_pose7(e.left.pose), _pose7(e.right.pose)] for e in states.eef])
    aperture = np.array([[e.left.gripper, e.right.gripper] for e in states.eef], float)
    objects = np.zeros((T, len(ids), 7))
    for t, scene in enumerate(states.scenes):
        for j, tid in enumerate(ids):
            obj = scene.get(tid)
            if obj is None:
                raise ValueError(f"seed {states.seed}: {tid} leaves the scene at frame {t}")
            objects[t, j] = _pose7(obj.pose)
    first = states.scenes[0].objects
    kind = [MOVABLE] * len(ids)
    colour = [colour_id(getattr(o, "colour", None)) for o in first]
    category = [category_id(CATEGORY_OF_ASSET.get(o.category)) for o in first]
    size = [tuple(o.size_xyz) for o in first]
    tracks = list(ids)
    if shelf_spec is not None:
        from oct_vla.data.entity_tokens import shelf_support_entities

        decks = shelf_support_entities(shelf_spec)
        for name, deck in zip(("lower_deck", "upper_deck"), decks):
            objects = np.concatenate([objects, np.tile(_pose7(deck.pose), (T, 1, 1))], axis=1)
            kind.append(FIXTURE)
            colour.append(colour_id(None))
            category.append(category_id(None))
            size.append(tuple(deck.size_xyz))
            tracks.append(name)
    wrist[..., 3:] = np.stack([Q.continuous(Q.normalize(wrist[:, a, 3:])) for a in range(2)], 1)
    objects[..., 3:] = np.stack([Q.continuous(Q.normalize(objects[:, j, 3:]))
                                 for j in range(objects.shape[1])], 1)
    annotations = {
        "source": "recording", "seed": states.seed, "success": states.success,
        "rule": states.metadata.get("rule"), "task": states.metadata.get("task"),
        "target_source": TARGET_SOURCE_LEGACY,
        "phases": states.phases,
        "targets": tuple(c.target_track_id for c in states.contexts),
        "physical": len(ids),
    }
    return Episode(np.asarray(states.timestamps, float), wrist, aperture, objects,
                   np.asarray(kind, np.int8), np.asarray(colour, np.int8),
                   np.asarray(category, np.int8), np.asarray(size, float), tuple(tracks),
                   annotations)


def read_recording(directory: str | Path, *, shelf: bool = True) -> Episode:
    """One recorded run as a physical ``Episode`` (states only)."""
    from oct_vla.data.store import read_states

    states = read_states(directory)
    spec = None
    if shelf and states.metadata.get("task", "").lower().startswith("shelf"):
        from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC

        spec = DEFAULT_SPEC
    return episode_from_states(states, shelf_spec=spec)
