"""The pose-window table contract (``pose-window-v1``).

Two objects, kept apart on purpose:

- :class:`Episode` is the **physical** trajectory on one clock: both wrists, apertures,
  every object's pose, and static per-object attributes. Synthetic episodes come from the
  generator, recorded ones from ``real.py``. Its ``annotations`` are evaluator-only (rule,
  eligible set, events, branch cue) and never reach a model.
- :class:`Table` is what the PFN **sees**: one row per timestamp, one column per entity
  (both wrists, objects, fixtures), each cell a pose7 with an observation mask, and a
  static header per column (kind, colour, category, size). A support table also carries
  the two goal columns (each wrist's pose at the next row's absolute timestamp); a query
  table never does, and stops at the query time t.

Semantic vocabularies are **stable**: red is the same id in every table. Column *order*
is shuffled independently per table, each column carrying its header and masks. A
registered variant relabels semantic ids with one mapping shared by a support and its
query (``relabel``); wrist side, entity kind, unknown and padding are never relabelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

SCHEMA_ID = "pose-window-v1"
HZ = 15.0
DT = 1.0 / HZ
POSE_DIM = 7
ARMS = ("left", "right")

#: Entity kinds. 0 is padding; the two wrists are distinct kinds (side is meaningful).
KINDS = ("pad", "left_wrist", "right_wrist", "movable", "fixture")
PAD, LEFT_WRIST, RIGHT_WRIST, MOVABLE, FIXTURE = range(len(KINDS))
WRIST_KINDS = (LEFT_WRIST, RIGHT_WRIST)
#: Stable semantic vocabularies; 0 is unknown in each.
COLOURS = ("unknown", "red", "blue", "green", "yellow")
CATEGORIES = ("unknown", "box", "can", "microphone", "plate", "container")
UNKNOWN = 0


def colour_id(name: str | None) -> int:
    return UNKNOWN if name is None else COLOURS.index(name)


def category_id(name: str | None) -> int:
    return UNKNOWN if name is None else CATEGORIES.index(name)


@dataclass
class Episode:
    """A physical trajectory sampled at ``HZ``. ``N`` objects (movables then fixtures)."""

    time: np.ndarray            # [T] seconds
    wrist: np.ndarray           # [T, 2, 7] left, right
    aperture: np.ndarray        # [T, 2] in [0, 1], 1 = open
    objects: np.ndarray         # [T, N, 7]
    kind: np.ndarray            # [N] MOVABLE or FIXTURE
    colour: np.ndarray          # [N] colour ids
    category: np.ndarray        # [N] category ids
    size: np.ndarray            # [N, 3] metres
    track_ids: tuple[str, ...]  # evaluator-only join keys
    annotations: dict = field(default_factory=dict)  # evaluator-only

    def __post_init__(self) -> None:
        T, N = len(self.time), len(self.track_ids)
        shapes = {"wrist": (T, 2, POSE_DIM), "aperture": (T, 2), "objects": (T, N, POSE_DIM),
                  "kind": (N,), "colour": (N,), "category": (N,), "size": (N, 3)}
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        for name in ("wrist", "objects"):
            q = getattr(self, name)[..., 3:]
            if not np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5):
                raise ValueError(f"{name} carries a non-unit quaternion")
        if np.any(np.isin(self.kind, WRIST_KINDS + (PAD,))):
            raise ValueError("episode objects are movables or fixtures; wrists live in `wrist`")

    @property
    def T(self) -> int:
        return len(self.time)


@dataclass
class Table:
    """What the model reads. ``C`` columns = 2 wrists + objects, in a shuffled order."""

    time: np.ndarray            # [T]
    pose: np.ndarray            # [T, C, 7]
    observed: np.ndarray        # [T, C] bool: a real measurement this row
    age: np.ndarray             # [T, C] s since the cell's last real measurement (0 if fresh)
    kind: np.ndarray            # [C]
    colour: np.ndarray          # [C]
    category: np.ndarray        # [C]
    size: np.ndarray            # [C, 3]
    goal: np.ndarray            # [T, 2, 7] each wrist's pose at the next row (support only)
    goal_valid: np.ndarray      # [T, 2] bool
    tracks: tuple[str, ...]     # evaluator-only: column -> track id ("left"/"right" for wrists)

    MODEL_FIELDS = ("time", "pose", "observed", "age", "kind", "colour", "category", "size",
                    "goal", "goal_valid")

    def model_inputs(self) -> dict[str, np.ndarray]:
        """Exactly the arrays a model may read; ``tracks`` and annotations stay out."""
        return {k: getattr(self, k) for k in self.MODEL_FIELDS}

    def column(self, track: str) -> int:
        return self.tracks.index(track)


def _columns(ep: Episode, rows: np.ndarray):
    pose = np.concatenate([ep.wrist[rows], ep.objects[rows]], axis=1)
    kind = np.concatenate([[LEFT_WRIST, RIGHT_WRIST], ep.kind]).astype(np.int8)
    colour = np.concatenate([[UNKNOWN, UNKNOWN], ep.colour]).astype(np.int8)
    category = np.concatenate([[UNKNOWN, UNKNOWN], ep.category]).astype(np.int8)
    size = np.concatenate([np.zeros((2, 3)), ep.size]).astype(np.float32)
    return pose.astype(np.float32), kind, colour, category, size, ("left", "right") + ep.track_ids


def _observation(ep: Episode, rows: np.ndarray, pose: np.ndarray, obs):
    """Apply an observation process ``obs = (pose [T, C, 7], observed [T, C], age [T, C])``
    over the episode's columns (``observation.py``); None means clean."""
    if obs is None:
        return pose, np.ones(pose.shape[:2], bool), np.zeros(pose.shape[:2], np.float32)
    o_pose, o_obs, o_age = obs
    return o_pose[rows].astype(np.float32), o_obs[rows], o_age[rows].astype(np.float32)


def support_table(ep: Episode, rows: np.ndarray | None = None, obs=None) -> Table:
    """A completed demonstration: chosen rows, each with both wrists' next-row poses as
    goals. ``rows`` defaults to every row; a goal is the wrist at the *next selected* row's
    absolute timestamp, so subsampling never shifts a target by a step."""
    rows = np.arange(ep.T) if rows is None else np.asarray(rows)
    if np.any(np.diff(rows) <= 0):
        raise ValueError("support rows must increase")
    pose, kind, colour, category, size, tracks = _columns(ep, rows)
    goal = np.zeros((len(rows), 2, POSE_DIM), np.float32)
    goal_valid = np.zeros((len(rows), 2), bool)
    goal[:-1] = ep.wrist[rows[1:]]
    goal_valid[:-1] = True
    pose, observed, age = _observation(ep, rows, pose, obs)
    return Table(ep.time[rows].astype(np.float32), pose, observed, age, kind, colour, category,
                 size, goal, goal_valid, tracks)


def query_table(ep: Episode, t: int, history: int = 16, obs=None) -> Table:
    """The observed prefix ending at row ``t`` (inclusive), at most ``history`` rows. No
    row after t enters, and the goal columns are masked placeholders."""
    if not 0 <= t < ep.T:
        raise IndexError(f"query time {t} outside the episode (T={ep.T})")
    rows = np.arange(max(0, t - history + 1), t + 1)
    pose, kind, colour, category, size, tracks = _columns(ep, rows)
    pose, observed, age = _observation(ep, rows, pose, obs)
    return Table(ep.time[rows].astype(np.float32), pose, observed, age, kind, colour, category,
                 size, np.zeros((len(rows), 2, POSE_DIM), np.float32),
                 np.zeros((len(rows), 2), bool), tracks)


def target_window(ep: Episode, t: int, horizon: int = 8,
                  cue: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Both wrists at t+1 .. t+H, sliced once from the physical track: ``[H, 2, 7]`` and a
    validity mask ``[H]``. Horizons past the episode end are masked. ``cue``: the first row
    at which a branch becomes observable (its state already differs there); a query before
    it is supervised only on rows before it, since nothing before the cue can tell the
    branches apart."""
    idx = t + 1 + np.arange(horizon)
    valid = idx < ep.T
    if cue is not None and t < cue:
        valid &= idx < cue
    window = np.zeros((horizon, 2, POSE_DIM), np.float32)
    window[valid] = ep.wrist[idx[valid]]
    return window, valid


def shuffle_columns(table: Table, rng: np.random.Generator) -> tuple[Table, np.ndarray]:
    """Permute entity columns; every column keeps its header, masks and track."""
    perm = rng.permutation(table.pose.shape[1])
    return replace(table, pose=table.pose[:, perm], observed=table.observed[:, perm],
                   age=table.age[:, perm], kind=table.kind[perm], colour=table.colour[perm],
                   category=table.category[perm], size=table.size[perm],
                   tracks=tuple(table.tracks[i] for i in perm)), perm


def semantic_relabeling(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One random relabeling of colour and category ids. Unknown (0) maps to itself."""
    out = {}
    for name, vocab in (("colour", COLOURS), ("category", CATEGORIES)):
        perm = np.arange(len(vocab))
        perm[1:] = 1 + rng.permutation(len(vocab) - 1)
        out[name] = perm
    return out


def relabel(table: Table, mapping: dict[str, np.ndarray]) -> Table:
    """Apply a semantic relabeling. Share one mapping between a support and its query;
    kinds, wrists, unknown and padding are untouched."""
    return replace(table, colour=mapping["colour"][table.colour].astype(np.int8),
                   category=mapping["category"][table.category].astype(np.int8))
