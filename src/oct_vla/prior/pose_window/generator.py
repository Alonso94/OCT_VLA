"""The analytic prior over paired episodes (plan stage 0.2; paper Sec. IV-A).

A **program** is a manipulation rule ``m`` crossed with a composition rule ``c``. For one
program, ``sample_episode`` draws a scene (layout, sizes, colours, categories, arm starts,
shelf geometry) and executes the program on it, producing a dense physical
:class:`schema.Episode`. A support and a query are two independent draws of the same
program; their object counts come from the arm's count matrix (``CountMatrix``).

Physical trajectory first: every motion is built on ``trajectory.Builder`` (one clock,
minimum-jerk + SLERP within a phase, rigid carry by a fixed wrist->object transform). The
observed table is made from it afterwards (``observation.py``).

Manipulation rules:
- ``carry_left`` / ``carry_right``: that arm picks and places every selected object;
- ``carry_nearest``: the arm on the object's side does;
- ``handover``: the arm on the object's side picks it and hands it to the other arm at a
  shared exchange pose; the receiver places it. The giver releases only once the receiver
  has closed on the object (reactive, as in ``branches.py``);
- ``push``: the arm on the object's side pushes it forward on the same shelf by ``PUSH_D``.

Composition rules:
- eligibility ``all`` | ``colour:<c>`` | ``category:<k>``; ineligible objects are
  distractors and never move;
- order ``left_to_right`` | ``right_to_left`` | ``small_to_large`` | ``large_to_small``;
- repeat ``until_empty`` | ``fixed_2`` (exactly two, then stop: "repeat three times" and
  "until empty" differ only when the counts do);
- compaction ``none`` | ``toward_neighbour`` (after each place, the other arm pushes the
  object along the row until the gap to its neighbour is ``COMPACT_GAP``; carries only).

Infeasible draws raise ``Infeasible`` and are re-drawn, never repaired; ``sample_episode``
counts rejections so they can be published per program and count.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import quat as Q
from .schema import (CATEGORIES, COLOURS, DT, FIXTURE, MOVABLE, Episode, category_id,
                     colour_id)
from .trajectory import (CLOSED, OPEN, TOP_DOWN, Builder, compose, invert, min_jerk,
                         orientation, ticks, wrist_for_grip)

MANIPULATIONS = ("carry_left", "carry_right", "carry_nearest", "handover", "push")
ELIGIBILITY_KINDS = ("all", "colour", "category")
ORDERS = ("left_to_right", "right_to_left", "small_to_large", "large_to_small")
REPEATS = ("until_empty", "fixed_2")
COMPACTIONS = ("none", "toward_neighbour")
#: Colours and categories the generator draws from (stable ids, schema vocabularies).
GEN_COLOURS = ("red", "blue", "green", "yellow")
GEN_CATEGORIES = ("box", "can")

MAX_ELIGIBLE = 5         # 5 only in the sealed extrapolation split
MAX_MOVABLES = 6
SOURCE_X = (-0.42, 0.42)
MIN_SEPARATION = 0.12    # m between object centres on the source shelf
PITCH_GAP = (0.03, 0.06)  # m initial gap left between placed neighbours (before compaction)
COMPACT_GAP = 0.01
PUSH_D = 0.10
#: The synthetic lower deck reaches 6 cm behind the objects and far enough forward that a
#: push stays on it (the RoboTwin shelf is shallower; the prior's geometry is broader).
LOWER_DEPTH = 0.24
LIFT = 0.10
CROSS_BODY = 0.30        # an arm may not work more than this past the midline
#: Arm speed scale per draw, so duration never identifies a rule.
SPEED = (0.8, 1.25)


class Infeasible(RuntimeError):
    """The program cannot run on this scene; the scene is re-drawn, never repaired."""


@dataclass(frozen=True)
class Program:
    manipulation: str
    eligibility: str          # "all", "colour:red", "category:can", ...
    order: str
    repeat: str
    compaction: str = "none"

    def __post_init__(self) -> None:
        kind = self.eligibility.split(":")[0]
        if self.manipulation not in MANIPULATIONS or kind not in ELIGIBILITY_KINDS \
                or self.order not in ORDERS or self.repeat not in REPEATS \
                or self.compaction not in COMPACTIONS:
            raise ValueError(f"not a program: {self}")
        if kind == "colour" and self.eligibility.split(":")[1] not in GEN_COLOURS:
            raise ValueError(f"unknown colour in {self.eligibility}")
        if kind == "category" and self.eligibility.split(":")[1] not in GEN_CATEGORIES:
            raise ValueError(f"unknown category in {self.eligibility}")
        if self.compaction != "none" and not self.manipulation.startswith("carry"):
            raise ValueError("compaction follows a placement; it applies to carries only")

    @property
    def eligibility_kind(self) -> str:
        return self.eligibility.split(":")[0]

    @property
    def family(self) -> tuple[str, str]:
        """(manipulation family, selection family): the unit the composition split holds out."""
        manip = "handover" if self.manipulation == "handover" else (
            "push" if self.manipulation == "push" else "carry")
        return manip, self.eligibility_kind

    @property
    def key(self) -> str:
        return "|".join((self.manipulation, self.eligibility, self.order, self.repeat,
                         self.compaction))


def all_programs() -> tuple[Program, ...]:
    elig = ["all"] + [f"colour:{c}" for c in GEN_COLOURS] + [f"category:{k}" for k in GEN_CATEGORIES]
    out = []
    for m, e, o, r, c in itertools.product(MANIPULATIONS, elig, ORDERS, REPEATS, COMPACTIONS):
        try:
            out.append(Program(m, e, o, r, c))
        except ValueError:
            continue
    return tuple(out)


# --- scenes -----------------------------------------------------------------------------

@dataclass
class Scene:
    xs: np.ndarray            # [n] source-shelf x
    yaws: np.ndarray          # [n]
    sizes: np.ndarray         # [n, 3]
    colours: list[str]
    categories: list[str]
    eligible: np.ndarray      # [n] bool
    lower: tuple[float, float]    # (y, top z)
    upper: tuple[float, float]    # (y, top z)
    homes: np.ndarray         # [2, 7]
    speed: float


def _size(category: str, rng) -> np.ndarray:
    if category == "box":
        return np.array([rng.uniform(0.04, 0.06), rng.uniform(0.04, 0.06), rng.uniform(0.06, 0.09)])
    d = rng.uniform(0.045, 0.06)
    return np.array([d, d, rng.uniform(0.08, 0.11)])


def sample_scene(program: Program, eligible: int, distractors: int,
                 rng: np.random.Generator, distractor_value: str | None = None) -> Scene:
    """``eligible`` objects the program may select plus ``distractors`` it must not.
    ``distractor_value``: give every distractor this colour/category (conflict pairs)."""
    n = eligible + distractors
    if n > MAX_MOVABLES or n < 1:
        raise Infeasible(f"{n} movables")
    kind = program.eligibility_kind
    value = program.eligibility.split(":")[1] if ":" in program.eligibility else None
    if kind != "all" and distractors == 0:
        # Without a distractor a selective rule is unobservable; keep it observable.
        raise Infeasible("a selective rule needs at least one distractor")
    if kind == "all" and distractors:
        raise Infeasible("'all' has no distractors")
    colours, categories = [], []
    for i in range(n):
        is_elig = i < eligible
        if kind == "colour":
            colours.append(value if is_elig else (distractor_value or rng.choice(
                [c for c in GEN_COLOURS if c != value])))
            categories.append(rng.choice(GEN_CATEGORIES))
        elif kind == "category":
            categories.append(value if is_elig else (distractor_value or rng.choice(
                [c for c in GEN_CATEGORIES if c != value])))
            colours.append(rng.choice(GEN_COLOURS))
        else:
            colours.append(rng.choice(GEN_COLOURS))
            categories.append(rng.choice(GEN_CATEGORIES))
    sizes = np.stack([_size(c, rng) for c in categories])
    if program.order in ("small_to_large", "large_to_small") and eligible > 1:
        # Spread eligible volumes by construction (>= 20 % apart) by scaling heights,
        # instead of rejecting most draws.
        idx = np.argsort(sizes[:eligible].prod(1))
        for a, b in zip(idx[:-1], idx[1:]):
            need = 1.2 * sizes[a].prod()
            if sizes[b].prod() < need:
                sizes[b, 2] = need / (sizes[b, 0] * sizes[b, 1])
        if sizes[:, 2].max() > 0.14:
            raise Infeasible("size spread needs objects too tall")
    xs = _spaced(n, rng)
    perm = rng.permutation(n)                     # eligibility independent of position
    lower = (rng.uniform(-0.27, -0.23), rng.uniform(0.72, 0.77))
    upper = (rng.uniform(-0.08, -0.04), rng.uniform(0.91, 0.96))
    homes = np.stack([np.concatenate([[-0.30 + rng.uniform(-0.04, 0.04),
                                       -0.12 + rng.uniform(-0.03, 0.03),
                                       1.20 + rng.uniform(-0.04, 0.04)], TOP_DOWN]),
                      np.concatenate([[0.30 + rng.uniform(-0.04, 0.04),
                                       -0.12 + rng.uniform(-0.03, 0.03),
                                       1.20 + rng.uniform(-0.04, 0.04)], TOP_DOWN])])
    return Scene(xs[perm], rng.uniform(-0.4, 0.4, n), sizes, colours, categories,
                 np.arange(n) < eligible, lower, upper, homes, rng.uniform(*SPEED))


def rescope(scene: Scene, program: Program) -> Scene:
    """The same physical scene, with eligibility recomputed for another program."""
    kind = program.eligibility_kind
    if kind == "all":
        elig = np.ones(len(scene.xs), bool)
    else:
        value = program.eligibility.split(":")[1]
        attr = scene.colours if kind == "colour" else scene.categories
        elig = np.array([a == value for a in attr])
    return Scene(scene.xs, scene.yaws, scene.sizes, scene.colours, scene.categories, elig,
                 scene.lower, scene.upper, scene.homes, scene.speed)


def _spaced(n: int, rng) -> np.ndarray:
    for _ in range(200):
        xs = np.sort(rng.uniform(*SOURCE_X, n))
        if n < 2 or np.diff(xs).min() >= MIN_SEPARATION:
            return xs
    raise Infeasible(f"cannot space {n} objects")


def selection(program: Program, scene: Scene) -> list[int]:
    """Indices the program moves, in order."""
    idx = [int(i) for i in np.flatnonzero(scene.eligible)]
    if program.order == "left_to_right":
        key = scene.xs
    elif program.order == "right_to_left":
        key = -scene.xs
    else:
        vol = scene.sizes.prod(1)
        key = vol if program.order == "small_to_large" else -vol
        vols = np.sort(vol[idx])
        # Size order must be visibly decidable: consecutive volumes differ by >= 15 %.
        if len(vols) > 1 and np.min(vols[1:] / vols[:-1]) < 1.15:
            raise Infeasible("volumes too close to order")
    idx = sorted(idx, key=lambda i: key[i])
    return idx[:2] if program.repeat == "fixed_2" else idx


# --- execution ---------------------------------------------------------------------------

def extent_x(size: np.ndarray, yaw: float) -> float:
    """Width of a yawed box's footprint along x."""
    return abs(np.cos(yaw)) * size[0] + abs(np.sin(yaw)) * size[1]


def _grasp_q(yaw: float) -> np.ndarray:
    return Q.multiply(Q.from_axis_angle([0, 0, 1], yaw), TOP_DOWN)


def _arm_for(program: Program, x: float) -> int:
    if program.manipulation == "carry_left":
        arm = 0
    elif program.manipulation == "carry_right":
        arm = 1
    else:
        arm = 0 if x < 0 else 1
    return arm


def _reachable(arm: int, x: float) -> bool:
    return x <= CROSS_BODY if arm == 0 else x >= -CROSS_BODY


class _Run:
    def __init__(self, program: Program, scene: Scene, rng) -> None:
        self.p, self.s, self.rng = program, scene, rng
        n = len(scene.xs)
        objs = np.stack([
            np.concatenate([[scene.xs[j], scene.lower[0], scene.lower[1] + scene.sizes[j, 2] / 2],
                            Q.from_axis_angle([0, 0, 1], scene.yaws[j])]) for j in range(n)])
        decks = np.stack([np.array([0.0, scene.lower[0] + LOWER_DEPTH / 2 - 0.06,
                                    scene.lower[1] - 0.01, *Q.IDENTITY]),
                          np.array([0.0, scene.upper[0], scene.upper[1] - 0.01, *Q.IDENTITY])])
        self.b = Builder(scene.homes, np.concatenate([objs, decks]))
        self.n = n
        self.placed: list[int] = []
        self.events: list[dict] = []

    def dur(self, base: float) -> float:
        return base * self.s.speed * self.rng.uniform(0.9, 1.1)

    def wrist_for_object(self, arm: int, obj_pose: np.ndarray) -> np.ndarray:
        """The wrist pose that puts the object ``arm`` holds at ``obj_pose`` (exact: the
        inverse of the attachment transform, never a nominal tool offset)."""
        _, rel = self.b.attached[arm]
        return compose(obj_pose, invert(rel))

    def _event(self, kind: str, obj: int, arm: int) -> None:
        self.events.append({"event": kind, "obj": f"obj_{obj}", "arm": arm, "row": self.b.now})

    def pick(self, arm: int, j: int) -> None:
        b = self.b
        obj = b.objects[-1][j]
        q = _grasp_q(self.s.yaws[j] + (np.pi / 2 if self.s.sizes[j, 0] > self.s.sizes[j, 1] else 0))
        grip = obj[:3].copy()          # recorded grasps hold the object at its centre
        b.segment(self.dur(1.0), {arm: wrist_for_grip(grip + [0, 0, LIFT], q)})
        b.segment(self.dur(0.5), {arm: wrist_for_grip(grip, q)})
        b.segment(0.3, aperture={arm: CLOSED})
        b.attach(arm, j)
        self._event("grasp", j, arm)
        b.segment(self.dur(0.5), {arm: wrist_for_grip(grip + [0, 0, LIFT], q)})

    def _slot(self, j: int) -> np.ndarray:
        """Next upper-shelf position: rows fill from the left, leaving a gap to compact."""
        s = self.s
        if not self.placed:
            x = -0.25
        else:
            prev = self.placed[-1]
            px = self.b.objects[-1][prev][0]
            x = px + (extent_x(s.sizes[prev], s.yaws[prev]) + extent_x(s.sizes[j], s.yaws[j])) / 2 \
                + self.rng.uniform(*PITCH_GAP)
        if x > 0.35:
            raise Infeasible("destination row overflows")
        return np.array([x, s.upper[0], s.upper[1] + s.sizes[j, 2] / 2])

    def place(self, arm: int, j: int) -> None:
        b = self.b
        dest = self._slot(j)
        if not _reachable(arm, dest[0]):
            raise Infeasible(f"arm {arm} cannot reach slot x={dest[0]:.2f}")
        upright = np.concatenate([dest, Q.from_axis_angle([0, 0, 1], self.s.yaws[j])])
        above = upright.copy()
        above[2] += LIFT
        b.segment(self.dur(1.2), {arm: self.wrist_for_object(arm, above)})
        b.segment(self.dur(0.5), {arm: self.wrist_for_object(arm, upright)})
        b.segment(0.3, aperture={arm: OPEN})
        retreat = b.wrist[-1][arm].copy()
        retreat[2] += LIFT
        b.detach(arm)
        self._event("place", j, arm)
        b.segment(self.dur(0.5), {arm: retreat})
        self.placed.append(j)

    def push(self, arm: int, j: int, direction: np.ndarray, distance: float) -> None:
        """Approach behind ``j`` opposite ``direction``, closed, and push it ``distance``."""
        b = self.b
        obj = b.objects[-1][j]
        direction = np.asarray(direction, float) / np.linalg.norm(direction)
        half = np.abs(Q.rotate(obj[3:], np.eye(3)) @ direction) @ self.s.sizes[j] / 2
        contact = obj[:3] - direction * (half + 0.015)
        contact[2] = obj[2]
        q = _grasp_q(np.arctan2(direction[1], direction[0]))
        b.segment(self.dur(0.8), {arm: wrist_for_grip(contact + [0, 0, LIFT], q)},
                  aperture={arm: CLOSED})
        b.segment(self.dur(0.5), {arm: wrist_for_grip(contact, q)})
        b.attach(arm, j)                           # kinematic push contact
        self._event("push_start", j, arm)
        end = contact + direction * distance
        b.segment(max(DT, self.dur(0.8) * distance / PUSH_D), {arm: wrist_for_grip(end, q)})
        b.detach(arm)
        self._event("push_end", j, arm)
        b.segment(self.dur(0.5), {arm: wrist_for_grip(end + [0, 0, LIFT], q)},
                  aperture={arm: OPEN})

    def compact(self, j: int, arm: int) -> None:
        if len(self.placed) < 2:
            return
        prev = self.placed[-2]
        s = self.s
        gap = (self.b.objects[-1][j][0] - self.b.objects[-1][prev][0]
               - (extent_x(s.sizes[j], s.yaws[j]) + extent_x(s.sizes[prev], s.yaws[prev])) / 2)
        if gap <= COMPACT_GAP:
            return
        if not _reachable(arm, self.b.objects[-1][j][0]):
            raise Infeasible("compacting arm cannot reach")
        self.push(arm, j, np.array([-1.0, 0.0, 0.0]), gap - COMPACT_GAP)

    def handover(self, giver: int, j: int, delay: dict | None = None) -> None:
        """Pick with ``giver``, exchange, place with the receiver (reactive giver)."""
        b = self.b
        receiver = 1 - giver
        self.pick(giver, j)
        side = -1.0 if giver == 0 else 1.0
        exchange = np.concatenate([[side * 0.04, -0.12, 1.05], b.objects[-1][j][3:]])
        b.segment(self.dur(1.0), {giver: self.wrist_for_object(giver, exchange)})
        obj = b.objects[-1][j]
        # The giver holds the object from the top; the receiver closes on its lower part
        # from its own side (tool z horizontal, pointing at the giver), clear of the
        # giver's fingers. Grasp frames: top-down vs side, compatible by construction.
        recv_q = orientation((side, 0.0, 0.0), (0.0, 1.0, 0.0))   # tool toward the giver
        low = obj[:3] - [0, 0, self.s.sizes[j, 2] / 4]
        recv_grasp = wrist_for_grip(low, recv_q)
        recv_pre = wrist_for_grip(low - [side * 0.10, 0, 0], recv_q)
        approach = ticks(self.dur(1.2))
        profile = np.concatenate([[0.0], min_jerk(approach)])
        start_r = b.wrist[-1][receiver].copy()
        progress, captured, state = 0.0, 0, "approach"
        t0 = b.now
        while True:
            w, a = b.wrist[-1].copy(), b.aperture[-1].copy()
            if state == "approach":
                rate = 1.0
                if delay and (b.now - t0) * DT >= delay["at"]:
                    rate = delay["slow"]
                progress = min(approach, progress + rate)
                lo = int(np.floor(progress))
                s = profile[lo] if lo >= approach else (
                    (1 - (progress - lo)) * profile[lo] + (progress - lo) * profile[lo + 1])
                w[receiver, :3] = start_r[:3] + s * (recv_pre[:3] - start_r[:3])
                w[receiver, 3:] = Q.slerp(start_r[3:], recv_pre[3:], np.array([s]))[0]
                if progress >= approach:
                    b.tick(w, a)
                    b.segment(self.dur(0.5), {receiver: recv_grasp})
                    b.segment(0.3, aperture={receiver: CLOSED})
                    state = "closed"
                    continue
            elif state == "closed":
                at = np.linalg.norm(b.wrist[-1][receiver, :3] - recv_grasp[:3]) < 0.01
                captured = captured + 1 if (at and b.aperture[-1][receiver] < 0.3) else 0
                if captured * DT >= 0.2:
                    break
            b.tick(w, a)
            if b.now - t0 > 2000:
                raise Infeasible("handover did not complete")
        b.segment(0.3, aperture={giver: OPEN})
        b.transfer(giver, receiver)
        self._event("transfer", j, receiver)
        back = b.wrist[-1][giver].copy()
        back[0] += side * 0.15
        b.segment(self.dur(0.6), {giver: back})
        self.place(receiver, j)

    def finish(self) -> None:
        b = self.b
        b.segment(self.dur(1.0), {0: self.s.homes[0], 1: self.s.homes[1]})
        b.hold(self.rng.uniform(0.8, 1.5))     # final dwell: a completed scene stays put


def execute(program: Program, scene: Scene, rng: np.random.Generator,
            perturb: dict | None = None) -> Episode:
    """Run ``program`` on ``scene``. ``perturb``: ``{"transfer": k, "at": s, "slow": r}`` delays
    the receiver of the k-th handover (``branches``)."""
    run = _Run(program, scene, rng)
    order = selection(program, scene)
    if not order:
        raise Infeasible("nothing to move")
    for k, j in enumerate(order):
        x = scene.xs[j]
        if program.manipulation == "push":
            arm = 0 if x < 0 else 1
            run.push(arm, j, np.array([0.0, 1.0, 0.0]), PUSH_D)
        elif program.manipulation == "handover":
            giver = 0 if x < 0 else 1
            delay = perturb if (perturb and perturb.get("transfer") == k) else None
            run.handover(giver, j, delay)
        else:
            arm = _arm_for(program, x)
            if not _reachable(arm, x):
                raise Infeasible(f"arm {arm} cannot reach x={x:.2f}")
            run.pick(arm, j)
            run.place(arm, j)
            if program.compaction == "toward_neighbour":
                run.compact(j, 1 - arm)
    run.finish()
    n = run.n
    ids = [f"obj_{j}" for j in range(n)] + ["lower_deck", "upper_deck"]
    sizes = np.concatenate([scene.sizes, [[0.84, LOWER_DEPTH, 0.02], [0.84, 0.12, 0.02]]])
    return run.b.episode(
        [MOVABLE] * n + [FIXTURE, FIXTURE],
        [colour_id(c) for c in scene.colours] + [0, 0],
        [category_id(c) for c in scene.categories] + [0, 0],
        sizes, ids,
        {"source": "synthetic", "program": program.key, "family": program.family,
         "selected": [f"obj_{j}" for j in order], "eligible_ids":
             [f"obj_{j}" for j in np.flatnonzero(scene.eligible)],
         "eligible_count": int(scene.eligible.sum()), "physical": n,
         "distractors": int(n - scene.eligible.sum()), "events": run.events,
         "perturb": perturb})


def sample_episode(program: Program, eligible: int, distractors: int,
                   rng: np.random.Generator, tries: int = 50,
                   rejections: dict | None = None) -> Episode:
    """Draw scenes until the program runs; count rejections by reason."""
    for _ in range(tries):
        try:
            scene = sample_scene(program, eligible, distractors, rng)
            return execute(program, scene, rng)
        except Infeasible as reason:
            if rejections is not None:
                key = str(reason).split(" x=")[0].split(" slot")[0]
                rejections[key] = rejections.get(key, 0) + 1
    raise Infeasible(f"{program.key}: no feasible scene in {tries} draws "
                     f"({eligible} eligible + {distractors} distractors)")


__all__ = ["CATEGORIES", "COLOURS", "Infeasible", "Program", "all_programs", "execute",
           "sample_episode", "sample_scene", "selection"]
