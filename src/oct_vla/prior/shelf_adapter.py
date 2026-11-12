"""Real shelf scenes in, KeyframePFN predictions out: the learned future-keyframe source.

PFN-VLA step 3. The frozen prior (``model.py``) was trained on synthetic entity tables
only. This module gives it a real scene in that same format and turns its prediction into
the ``[3, 24]`` future-keyframe rows the policy's second KV branch reads
(``data/future_keyframes.py``), or into its latent query cells for the additional
latent arm (``paper/3_pfn_vla_interface.md`` §5).

**Inputs.** The query uses only what the policy observes:
- the tracked scene: object poses and sizes, keyed by perception's track ids;
- the end-effector state;
- the static support geometry;
- the episode's first frame, kept by the caller.

The task manager's ``TaskContext`` (target, neighbour, phase) never enters the query.
``query`` does not take one, and a test asserts this. Support demonstrations are
recordings, so their keyframes come from the recording's phase labels
(``data/future_keyframes.interaction_events``), as the plan's G1b specifies.

**Column identity.** ``build_entity_tokens`` sorts movables by their current geometry,
so an object's column changes as it moves. The prior attends down each column over
time, so here a column is a track: movables are ordered by track id on every row of a
table. The prior is equivariant to column order, so that order carries no information.

**Conventions bridged.**
- *Gripper reference point.* The prior's gripper point sits about 0.02 m below a
  grasped object's top (``generator.py``). The recorded EEF sits 0.108 m above the
  top at every grasp of the v4 leftmost corpus (q05–q95 0.106–0.110 m). Gripper
  heights are therefore shifted down by ``EEF_TO_GRIP`` on the way in and back up on
  the way out, so the output rows are in the EEF convention the oracle rows use.
- *Size range.* Shelf objects are slightly larger than the generator's: up to
  0.078 m wide (against 0.06) and 0.097 m tall (against 0.08). Both decks lie inside
  its support ranges. G1b measures what this costs.
- *What the prior cannot say.* It predicts no gripper orientation, so the gripper row
  carries the identity rotation. It has no "done" output, so the mask is always on;
  the policy learns what a prediction means after the last transfer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene
from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM, _token
from oct_vla.data.future_keyframes import FUTURE_DIM, FUTURE_EVENTS, FUTURE_ROWS, _row

from . import programs as P
from .batches import LANG_AMBIGUOUS, LANG_NONE
from .generator import K_MAX, N_MAX, S_MAX
from .schema import (
    EVENTS,
    GRASP,
    INITIAL,
    N_FEATURES,
    PLACE,
    POS,
    SIZE,
    from_entity_tokens,
)

#: Height of the recorded EEF above the prior's gripper point (see the module docstring).
EEF_TO_GRIP = 0.128
#: The instruction the prior is given: ambiguous, so the support carries the rule.
INSTRUCTION = P.AMBIGUOUS_INSTRUCTIONS[0]
#: Support demonstrations per query. The prior trained on 1..K_MAX.
K_SUPPORT = 3
_SIDES = ("left", "right")
_EVENT_INDEX = {"grasp": GRASP, "place": PLACE}


def _table(scene: ObjectScene, eef, supports: Sequence, order: Sequence[str]):
    """One row of a table: features [N_MAX, 9] and types [N_MAX] (-1 padding)."""
    n = len(order) + 2 + len(supports)
    if n > N_MAX:
        raise ValueError(f"{n} entities exceed the prior's {N_MAX} columns")
    tokens = []
    for track_id in order:
        item = scene.get(track_id)
        if item is None:
            raise ValueError(f"track {track_id!r} left the scene; the table cannot follow it")
        tokens.append(_token(item.pose, item.size_xyz, 0.0, "movable"))
    tokens += [_token(eef.left.pose, (0, 0, 0), eef.left.gripper, "left_gripper"),
               _token(eef.right.pose, (0, 0, 0), eef.right.gripper, "right_gripper")]
    tokens += [_token(s.pose, s.size_xyz, 0.0, "support") for s in supports]
    feats, types = from_entity_tokens(np.asarray(tokens), np.ones(n, bool))
    feats[len(order):len(order) + 2, 2] -= EEF_TO_GRIP
    out_f = np.zeros((N_MAX, N_FEATURES), np.float32)
    out_t = np.full(N_MAX, -1, np.int8)
    out_f[:n], out_t[:n] = feats, types
    return out_f, out_t


@dataclass(frozen=True)
class Query:
    """The prior's query: initial and current rows [2, N_MAX, 9], types, column -> track."""

    feats: np.ndarray
    types: np.ndarray
    order: tuple[str, ...]


def query(initial_scene: ObjectScene, initial_eef, scene: ObjectScene, eef,
          supports: Sequence) -> Query:
    """The query for one step. Takes no TaskContext: nothing task-specific may enter."""
    order = tuple(sorted(o.track_id for o in initial_scene.objects))
    first, types = _table(initial_scene, initial_eef, supports, order)
    current, types_now = _table(scene, eef, supports, order)
    if not np.array_equal(types, types_now):
        raise ValueError("the entity set changed since the first frame")
    return Query(np.stack([first, current]), types, order)


@dataclass(frozen=True)
class SupportDemo:
    """One demonstration as a keyframe table: the initial frame and every grasp and place."""

    feats: np.ndarray  # [S_MAX, N_MAX, 9]
    types: np.ndarray  # [N_MAX]
    rows: np.ndarray  # [S_MAX] bool
    event: np.ndarray  # [S_MAX]
    gripper: np.ndarray  # [S_MAX], -1 on the initial row
    flag: np.ndarray  # [S_MAX, N_MAX] acted object and acting gripper
    seed: int


def support_demo(episode, supports: Sequence) -> SupportDemo:
    """A recorded demonstration as a support table, keyframes from its phase labels."""
    from oct_vla.data.future_keyframes import interaction_events

    samples = episode.samples
    order = tuple(sorted(o.track_id for o in samples[0].scene.objects))
    keyframes = [(0, INITIAL, -1, None)] + [
        (e.frame, _EVENT_INDEX[e.kind], _SIDES.index(e.side), e.track_id)
        for e in interaction_events(episode)]
    if len(keyframes) > S_MAX:
        raise ValueError(f"seed {episode.seed}: {len(keyframes)} keyframes exceed {S_MAX} rows")
    feats = np.zeros((S_MAX, N_MAX, N_FEATURES), np.float32)
    rows = np.zeros(S_MAX, bool)
    event = np.zeros(S_MAX, np.int8)
    gripper = np.full(S_MAX, -1, np.int8)
    flag = np.zeros((S_MAX, N_MAX), bool)
    types = None
    for r, (frame, kind, side, track) in enumerate(keyframes):
        sample = samples[frame]
        feats[r], row_types = _table(sample.scene, sample.observation.eef, supports, order)
        if types is not None and not np.array_equal(types, row_types):
            raise ValueError(f"seed {episode.seed}: the entity set changed at frame {frame}")
        types = row_types
        rows[r], event[r], gripper[r] = True, kind, side
        if track is not None:
            flag[r, order.index(track)] = True
            flag[r, len(order) + side] = True
    return SupportDemo(feats, types, rows, event, gripper, flag, int(episode.seed))


def pack(demos: Sequence[SupportDemo], queries: Sequence[Query], lang: bool = True) -> dict:
    """A batch of queries, all with the same support demos, in the prior's input format."""
    if len(demos) > K_MAX:
        raise ValueError(f"{len(demos)} support demos exceed the prior's {K_MAX}")
    b = len(queries)
    ctx = {
        "ctx_feats": np.zeros((K_MAX, S_MAX, N_MAX, N_FEATURES), np.float32),
        "ctx_types": np.full((K_MAX, N_MAX), -1, np.int8),
        "ctx_rows": np.zeros((K_MAX, S_MAX), bool),
        "ctx_event": np.zeros((K_MAX, S_MAX), np.int8),
        "ctx_gripper": np.full((K_MAX, S_MAX), -1, np.int8),
        "ctx_flag": np.zeros((K_MAX, S_MAX, N_MAX), bool),
        "demo_valid": np.zeros(K_MAX, bool),
    }
    for d, demo in enumerate(demos):
        ctx["ctx_feats"][d], ctx["ctx_types"][d] = demo.feats, demo.types
        ctx["ctx_rows"][d], ctx["ctx_event"][d] = demo.rows, demo.event
        ctx["ctx_gripper"][d], ctx["ctx_flag"][d] = demo.gripper, demo.flag
        ctx["demo_valid"][d] = True
    batch = {k: np.repeat(v[None], b, 0) for k, v in ctx.items()}
    batch["q_feats"] = np.stack([q.feats for q in queries]).astype(np.float32)
    batch["q_types"] = np.stack([q.types for q in queries])
    kind = LANG_AMBIGUOUS if lang else LANG_NONE
    batch["lang_kind"] = np.full(b, kind, np.int64)
    batch["lang_index"] = np.full(b, P.ALL_INSTRUCTIONS.index(INSTRUCTION) if lang else -1,
                                  np.int64)
    return batch


def infer(model, batch: dict, lang_table, device="cpu", latent: bool = False) -> dict:
    """The prior's heads in metric units (``model.to_metric``), and, if asked, the
    current-row query cells ``latent`` [B, N_MAX+1, W] with their mask ``latent_ok``."""
    import torch

    from .model import to_metric, to_torch

    model.eval()
    with torch.no_grad():
        t = to_torch(batch, lang_table, device)
        out = to_metric(model(t, teacher=False))
        if latent:
            keys, key_ok = model.encode_context(t)
            cells, ok = model.query_cells(t, keys, key_ok)
            out["latent"] = cells[:, 1].float().cpu().numpy()
            out["latent_ok"] = ok[:, 1].cpu().numpy()
    return out


def _yaw_pose(position, sin_cos) -> Pose:
    yaw = float(np.arctan2(sin_cos[0], sin_cos[1]))
    return Pose(tuple(float(v) for v in position), (0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)),
                WORKCELL_FRAME)


def keyframe_rows(out: dict, i: int, q: Query) -> tuple[np.ndarray, np.ndarray]:
    """Prediction ``i`` of ``infer`` -> future-keyframe rows [3, 24] and mask [3].

    An interaction prior fills all three rows. A goal-only prior predicts the object and
    its goal but no gripper or event, so its gripper row is masked out and its event
    one-hot is left empty: rows the policy cannot mistake for an interaction prediction.
    """
    col = int(out["pointer"][i].argmax())
    size = tuple(float(v) for v in q.feats[1, col, SIZE])
    goal = _token(_yaw_pose(out["final"][i][:3], out["final"][i][3:5]), size, 0.0, "movable")
    if "event" not in out:
        current = q.feats[1, col]
        obj = _token(_yaw_pose(current[POS], current[3:5]), size, 0.0, "movable")
        rows = np.zeros((FUTURE_ROWS, FUTURE_DIM), np.float32)
        for r, token in ((0, obj), (2, goal)):
            rows[r, :ENTITY_TOKEN_DIM] = token
            rows[r, ENTITY_TOKEN_DIM + r] = 1.0
        return rows, np.array([True, False, True])
    event = EVENTS[int(out["event"][i].argmax()) + 1]
    if event not in FUTURE_EVENTS:
        raise ValueError(f"event {event!r} is outside the future-keyframe vocabulary")
    side = int(out["gripper"][i].argmax())
    obj = _token(_yaw_pose(out["obj"][i][:3], out["obj"][i][3:5]), size, 0.0, "movable")
    grip_pos = np.asarray(out["grip_pos"][i], np.float64) + (0.0, 0.0, EEF_TO_GRIP)
    grip = _token(Pose(tuple(grip_pos), (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME), (0, 0, 0),
                  float(out["aperture"][i]), f"{_SIDES[side]}_gripper")
    rows = np.stack([_row(obj, "object_keyframe", event), _row(grip, "gripper_keyframe", event),
                     _row(goal, "object_goal", event)])
    return rows, np.ones(FUTURE_ROWS, bool)


def predicted_track(out: dict, i: int, q: Query) -> str:
    """The track the prior says acts next."""
    return q.order[int(out["pointer"][i].argmax())]


def load_prior(run_dir, device="cpu"):
    """A frozen prior from ``scripts/prior_train.py`` output."""
    import torch

    from .model import KeyframePFN

    blob = torch.load(f"{run_dir}/model.pt", map_location="cpu")
    model = KeyframePFN(**blob["config"])
    model.load_state_dict(blob["state"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval()
