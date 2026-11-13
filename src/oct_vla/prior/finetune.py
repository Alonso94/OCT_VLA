"""Fine-tuning items from recorded demonstrations: PFN-VLA stage B.

The prior is pretrained on synthetic interaction primitives (stage A). Stage B
fine-tunes it on the object-centric part of the real demonstrations the VLA trains on,
before the VLA reads it (stage C). This closes the sim gap G1b measured:
- the right arm's compaction push, which the prior had no vocabulary for;
- the shelf's placement geometry;
- the prior's "leftmost first" default.

Three guards keep stage B from destroying what stage A bought
(``paper/3_pfn_vla_interface.md``):
1. **Several rules, never one.** A query's support is drawn from demonstrations of its
   own rule. Frame-0 *conflict pairs* put one scene under both rules' supports with
   both rules' targets, so a prior that ignores its support cannot fit both. On one
   rule alone it would learn that rule and stop reading the support, GEMIL's failure.
2. **Synthetic replay** (the training script mixes generator batches in), so the
   pretrained prior is kept, not overwritten.
3. **Held-out data stays held out.** Only training seed blocks are read
   (``docs/data_protocol.md``: 100–199 and 350–399); validation (200–249) is G1b's;
   the reserved blocks are refused. The held-out middle-first rule has no recordings
   here at all.

**Targets** follow the generator's conventions, through the same adapter as inference
(``shelf_adapter``):
- the next event's object column, acting gripper and kind (grasp / place);
- the object's pose at that event;
- the acting gripper's point (EEF shifted by ``EEF_TO_GRIP``) and aperture there;
- the object's pose when it is finally placed.

A conflict partner's aperture and final pose were never recorded, so they carry weight
0 (``model.loss``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import shelf_adapter as A
from .schema import GRASP, PLACE, POS, SIZE

TRAIN_BLOCKS = ((100, 200), (350, 400))
VAL_BLOCK = (200, 250)
RESERVED_BLOCKS = ((250, 350), (450, 550), (650, 750))
#: Rule -> how it picks the first object of a fresh scene (all on the lower shelf).
FIRST_OBJECT = {"leftmost": np.argmin, "rightmost": np.argmax}
#: The acting arm on the shelf (oracle/arms.py: the mover is always the left arm).
MOVER = 0
#: The prior's gripper point at a grasp sits this far below the object's top
#: (generator.py), and so does the shifted real EEF (shelf_adapter.EEF_TO_GRIP).
GRIP_BELOW_TOP = 0.02

KEYS = ("ctx_feats", "ctx_types", "ctx_rows", "ctx_event", "ctx_gripper", "ctx_flag",
        "demo_valid", "q_feats", "q_types", "t_obj", "t_grip", "t_event", "t_obj_pose",
        "t_grip_pos", "t_aperture", "t_final", "lang_kind", "lang_index",
        "m_obj", "m_grip_pos", "m_aperture", "m_final")


def split_of(seed: int) -> str:
    """The protocol's split of a three-object seed; refuses the reserved blocks."""
    if any(lo <= seed < hi for lo, hi in RESERVED_BLOCKS):
        raise ValueError(f"seed {seed} is in a reserved block; no code may read it")
    if any(lo <= seed < hi for lo, hi in TRAIN_BLOCKS):
        return "train"
    if VAL_BLOCK[0] <= seed < VAL_BLOCK[1]:
        return "val"
    raise ValueError(f"seed {seed} is in no three-object split")


def subset(episodes: list, n: int | None, seed: int, rule: str) -> list:
    """``n`` of one rule's training episodes, the same for a given (seed, rule).

    The low-data curve (G1c) trains on these. Drawn from training seeds only, so a
    subset can never reach validation, and returned in seed order so the selection
    does not depend on directory listing."""
    import zlib

    if n is None or n >= len(episodes):
        return sorted(episodes, key=lambda e: int(e.seed))
    if n < 1:
        raise ValueError("a subset needs at least one demonstration per rule")
    for ep in episodes:
        if split_of(int(ep.seed)) != "train":
            raise ValueError(f"seed {ep.seed} is not a training seed")
    ordered = sorted(episodes, key=lambda e: int(e.seed))
    rng = np.random.default_rng([seed, zlib.crc32(rule.encode())])
    chosen = sorted(rng.choice(len(ordered), n, replace=False))
    return [ordered[i] for i in chosen]


@dataclass
class Item:
    query: A.Query
    target: dict
    rule: str
    episode: int  # index into that rule's episodes, so a support never shows its own query


@dataclass
class RealData:
    """Items and support demos per rule, plus frame-0 conflict pairs."""

    items: dict[str, list[Item]] = field(default_factory=dict)
    demos: dict[str, list[A.SupportDemo]] = field(default_factory=dict)
    pairs: list[tuple[Item, Item]] = field(default_factory=list)


def _pose5(row) -> np.ndarray:
    return np.asarray(row[:5], np.float32)


def episode_items(episode, supports, rule: str, index: int, stride: int = 1) -> list[Item]:
    """Every ``stride``-th frame with a next grasp or place, and its target."""
    from oct_vla.data.future_keyframes import interaction_events

    samples = episode.samples
    s0 = samples[0]
    order = tuple(sorted(o.track_id for o in s0.scene.objects))
    n = len(order)
    events = interaction_events(episode)
    placed = {e.track_id: e.frame for e in events if e.kind == "place"}

    def table(frame):
        s = samples[frame]
        return A._table(s.scene, s.observation.eef, supports, order)[0]

    at = {e.frame: table(e.frame) for e in events}
    items = []
    k = 0
    for t in range(0, len(samples), stride):
        while k < len(events) and events[k].frame <= t:
            k += 1
        if k == len(events):
            break
        e = events[k]
        col, side = order.index(e.track_id), ("left", "right").index(e.side)
        here, final = at[e.frame], at[placed[e.track_id]]
        target = {
            "t_obj": col, "t_grip": side, "t_event": GRASP if e.kind == "grasp" else PLACE,
            "t_obj_pose": _pose5(here[col]), "t_grip_pos": here[n + side, POS].copy(),
            "t_aperture": np.float32(here[n + side, 8]), "t_final": _pose5(final[col]),
            "m_obj": 1.0, "m_grip_pos": 1.0, "m_aperture": 1.0, "m_final": 1.0,
        }
        q = A.query(s0.scene, s0.observation.eef, samples[t].scene, samples[t].observation.eef,
                    supports)
        items.append(Item(q, target, rule, index))
    return items


def frame0_target(q: A.Query, rule: str) -> dict:
    """The next keyframe of a fresh scene under ``rule``: grasp its first object with the
    mover. Aperture and final pose are not determined without the recording, so they are
    marked missing."""
    movable = np.flatnonzero(q.types == 0)
    first = movable[int(FIRST_OBJECT[rule](q.feats[1, movable, 0]))]
    obj = q.feats[1, first]
    grip = obj[POS].copy()
    grip[2] += obj[SIZE][2] / 2 - GRIP_BELOW_TOP
    return {"t_obj": int(first), "t_grip": MOVER, "t_event": GRASP, "t_obj_pose": _pose5(obj),
            "t_grip_pos": grip.astype(np.float32), "t_aperture": np.float32(0.0),
            "t_final": _pose5(obj), "m_obj": 1.0, "m_grip_pos": 1.0, "m_aperture": 0.0,
            "m_final": 0.0}


def build(episodes_by_rule: dict, supports_by_rule: dict, stride: int = 1) -> RealData:
    """Training items from training-block episodes only."""
    data = RealData()
    rules = sorted(episodes_by_rule)
    if len(rules) < 2:
        raise ValueError("fine-tuning on one rule teaches the prior to ignore its support")
    for rule in rules:
        episodes = episodes_by_rule[rule]
        for ep in episodes:
            if split_of(int(ep.seed)) != "train":
                raise ValueError(f"seed {ep.seed} is not a training seed")
        supports = supports_by_rule[rule]
        data.demos[rule] = [A.support_demo(ep, supports) for ep in episodes]
        data.items[rule] = [it for i, ep in enumerate(episodes)
                            for it in episode_items(ep, supports, rule, i, stride)]
    for rule in rules:
        for item in data.items[rule]:
            if not np.array_equal(item.query.feats[0], item.query.feats[1]):
                continue  # frame 0 only: both rules' next keyframe is defined there
            for other in rules:
                if other != rule:
                    partner = Item(item.query, frame0_target(item.query, other), other, -1)
                    data.pairs.append((item, partner))
    return data


def _packed(item: Item, demos: list) -> dict:
    b = A.pack(demos, [item.query])
    for key, value in item.target.items():
        b[key] = np.asarray([value])
    return b


def sample(data: RealData, rng: np.random.Generator, size: int, k_range=(1, 3),
           pair_fraction: float = 0.25) -> dict:
    """``size`` real items: some frame-0 conflict pairs (adjacent), the rest single items,
    each with 1..3 support demos of its own rule (never its own episode)."""
    parts = []
    rules = sorted(data.items)

    def support(rule, exclude):
        pool = [d for i, d in enumerate(data.demos[rule]) if i != exclude]
        k = min(int(rng.integers(k_range[0], k_range[1] + 1)), len(pool))
        return [pool[j] for j in rng.choice(len(pool), k, replace=False)]

    n_pairs = int(size * pair_fraction) // 2 if data.pairs else 0
    for j in rng.choice(len(data.pairs), n_pairs) if n_pairs else []:
        own, partner = data.pairs[j]
        # The same supports' count, from each rule; the partner's never include the
        # query's episode either (it belongs to the other rule).
        parts.append(_packed(own, support(own.rule, own.episode)))
        parts.append(_packed(partner, support(partner.rule, -1)))
    while sum(len(p["t_obj"]) for p in parts) < size:
        rule = rules[int(rng.integers(len(rules)))]
        item = data.items[rule][int(rng.integers(len(data.items[rule])))]
        parts.append(_packed(item, support(rule, item.episode)))
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in KEYS}


def with_masks(batch: dict) -> dict:
    """A synthetic generator batch in ``KEYS``, every target present."""
    n = len(batch["t_obj"])
    out = {k: batch[k] for k in KEYS if not k.startswith("m_")}
    out.update({k: np.ones(n, np.float32) for k in ("m_obj", "m_grip_pos", "m_aperture",
                                                    "m_final")})
    return out


def concat(*batches: dict) -> dict:
    return {k: np.concatenate([np.asarray(b[k]).astype(np.asarray(batches[0][k]).dtype)
                               for b in batches]) for k in KEYS}
