"""RoboTwin observations -> the measured history as a growing physical ``Episode``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from oct_vla.bpfn import quat
from oct_vla.bpfn.data.real_data.robotwin import SHELF_DECKS
from oct_vla.bpfn.schema import FIXTURE, MOVABLE, Episode, category_id, colour_id


def _pose7(pose) -> np.ndarray:
    return np.array([*pose.position, *pose.orientation], float)


@dataclass
class History:
    """Measured rows since reset, as a growing physical ``Episode`` (wrists, objects in
    first-seen track order, shelf decks as fixtures)."""

    times: list = field(default_factory=list)
    wrists: list = field(default_factory=list)
    apertures: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    tracks: tuple = ()
    header: dict = field(default_factory=dict)

    def add(self, obs) -> None:
        if not self.tracks:
            objs = obs.scene.objects
            self.tracks = tuple(o.track_id for o in objs)
            self.header = {
                "kind": [MOVABLE] * len(objs) + [FIXTURE] * len(SHELF_DECKS),
                "colour": [colour_id(o.colour) for o in objs] + [0] * len(SHELF_DECKS),
                "category": [category_id(_category(o.category)) for o in objs] + [0] * len(SHELF_DECKS),
                "size": [tuple(o.size_xyz) for o in objs] + [size for _, _, size in SHELF_DECKS],
                "decks": [np.array([*centre, 0.0, 0.0, 0.0, 1.0]) for _, centre, _ in SHELF_DECKS],
            }
        by_id = {o.track_id: o for o in obs.scene.objects}
        row = [(_pose7(by_id[t].pose) if t in by_id else self.objects[-1][i])
               for i, t in enumerate(self.tracks)]
        self.objects.append(np.array(row + self.header["decks"]))
        self.wrists.append(np.stack([_pose7(obs.eef.left.pose), _pose7(obs.eef.right.pose)]))
        self.apertures.append([obs.eef.left.gripper, obs.eef.right.gripper])
        self.times.append(float(obs.timestamp))

    def episode(self) -> Episode:
        wr = np.stack(self.wrists)
        wr[..., 3:] = np.stack([quat.continuous(quat.normalize(wr[:, a, 3:])) for a in range(2)], 1)
        ob = np.stack(self.objects)
        ob[..., 3:] = np.stack([quat.continuous(quat.normalize(ob[:, j, 3:])) for j in range(ob.shape[1])], 1)
        h = self.header
        return Episode(np.asarray(self.times), wr, np.asarray(self.apertures, float), ob,
                       np.asarray(h["kind"], np.int8), np.asarray(h["colour"], np.int8),
                       np.asarray(h["category"], np.int8), np.asarray(h["size"], float),
                       self.tracks + tuple(name for name, _, _ in SHELF_DECKS), {"source": "live"})


def _category(asset: str | None) -> str | None:
    from oct_vla.bpfn.data.real_data.robotwin import CATEGORY_OF_ASSET

    return CATEGORY_OF_ASSET.get(asset)
