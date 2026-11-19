"""Reactive handover branches from a common prefix (plan stage 0.2).

One giver carries an object to an exchange pose and holds it there. The receiver
approaches, descends and closes. The giver opens only once the receiver is at its grasp
pose with a closed gripper for ``CAPTURE_DWELL``; then the object transfers rigidly and
the giver retreats. These decisions are made every tick from the *physical state*, so
nothing is scripted in advance.

Branches share every tick until the perturbation starts:
- ``nominal``: the receiver approaches at its planned rate;
- ``delayed``: from ``delay_at`` the receiver's approach runs at ``slow`` of its rate;
- ``sham``: the receiver pauses briefly, then catches up before reaching its pre-grasp,
  so the exchange happens when it does in ``nominal`` (within a tick): an observed event
  with no effect on the giver.

``cue`` is the first row at which a branch's observed state differs from ``nominal``.
Before it, inputs are identical across branches, and ``schema.target_window(..., cue=)``
masks every horizon past it, so no supervised window asks the model to anticipate a
branch it cannot see.
"""

from __future__ import annotations

import numpy as np

from . import quat as Q
from .schema import DT, FIXTURE, MOVABLE, Episode, category_id, colour_id
from .trajectory import (CLOSED, OPEN, TOP_DOWN, Builder, min_jerk, orientation, ticks,
                         wrist_for_grip)

BRANCHES = ("nominal", "delayed", "sham")
CAPTURE_RADIUS = 0.01   # m: receiver wrist within this of its grasp pose
CAPTURE_DWELL = 0.2     # s closed at the grasp pose before the giver may open
CLOSED_BELOW = 0.3      # aperture below which a gripper counts as closed
#: Side grasps: the tool axis horizontal, along +x or -x.
SIDE_PLUS_X = orientation((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
SIDE_MINUS_X = orientation((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
OBJECT_SIZE = np.array([0.04, 0.04, 0.16])


def handover(branch: str = "nominal", giver: int = 0, delay_at: float = 1.0,
             slow: float = 0.35, sham_pause: float = 0.25) -> Episode:
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}")
    receiver = 1 - giver
    side = -1.0 if giver == 0 else 1.0          # giver starts on its own side
    giver_q = SIDE_PLUS_X if giver == 0 else SIDE_MINUS_X
    exchange = np.array([0.0, -0.10, 1.05])     # object centre at the exchange
    obj0 = np.array([side * 0.25, -0.10, 1.05, *Q.IDENTITY])
    giver_start = wrist_for_grip(obj0[:3], giver_q)
    giver_at_exchange = wrist_for_grip(exchange, giver_q)
    recv_grasp = wrist_for_grip(exchange + [0, 0, 0.04], TOP_DOWN)
    recv_pre = wrist_for_grip(exchange + [0, 0, 0.14], TOP_DOWN)
    recv_start = np.concatenate([[-side * 0.30, -0.12, 1.25], TOP_DOWN])
    place = np.array([-side * 0.25, -0.06, 1.0])
    fixture = np.array([0.0, -0.06, 0.90, *Q.IDENTITY])  # the destination deck

    wrists0 = np.zeros((2, 7))
    wrists0[giver], wrists0[receiver] = giver_start, recv_start
    b = Builder(wrists0, np.stack([obj0, fixture]), aperture0=[OPEN, OPEN])
    b.wrist[0][giver] = giver_start
    b.aperture[0][giver] = CLOSED
    b.attach(giver, 0)

    # Receiver approach (start -> pre-grasp) as progress along a minimum-jerk path,
    # advanced per tick; the branch changes only the rate.
    approach_ticks = ticks(1.6)
    profile = np.concatenate([[0.0], min_jerk(approach_ticks)])
    progress = 0.0                                # in ticks along the nominal profile
    giver_carry = ticks(1.0)
    descend, close, opening, retreat, carry = ticks(0.5), ticks(0.3), ticks(0.3), ticks(0.6), ticks(1.2)
    state_g, state_r = "carry", "approach"
    k_g = k_r = 0
    captured_for = 0
    release_row = transfer_row = None
    t = 0
    start = {"g": b.wrist[-1][giver].copy(), "r": b.wrist[-1][receiver].copy()}

    def lerp_pose(a, c, s):
        return np.concatenate([a[:3] + s * (c[:3] - a[:3]),
                               Q.slerp(a[3:], c[3:], np.array([s]))[0]])

    while True:
        t += 1
        now = t * DT
        w = b.wrist[-1].copy()
        a = b.aperture[-1].copy()
        # --- receiver ---
        if state_r == "approach":
            rate = 1.0
            if branch == "delayed" and now >= delay_at:
                rate = slow
            if branch == "sham" and delay_at <= now < delay_at + sham_pause:
                rate = 0.0
            if branch == "sham" and now >= delay_at + sham_pause:
                # Recover the paused progress over the approach time that remains.
                remaining = approach_ticks * DT - (delay_at + sham_pause)
                rate = 1.0 + sham_pause / max(DT, remaining)
            progress = min(approach_ticks, progress + rate)
            lo = int(np.floor(progress))
            frac = progress - lo
            s = profile[lo] if lo >= approach_ticks else (1 - frac) * profile[lo] + frac * profile[lo + 1]
            w[receiver] = lerp_pose(start["r"], recv_pre, s)
            if progress >= approach_ticks:
                state_r, k_r, start["r"] = "descend", 0, w[receiver].copy()
        elif state_r == "descend":
            k_r += 1
            w[receiver] = lerp_pose(start["r"], recv_grasp, min_jerk(descend)[k_r - 1])
            if k_r == descend:
                state_r, k_r = "close", 0
        elif state_r == "close":
            k_r += 1
            a[receiver] = OPEN + (CLOSED - OPEN) * k_r / close
            if k_r == close:
                state_r = "wait"
        elif state_r == "wait":
            if transfer_row is not None:
                state_r, k_r, start["r"] = "carry", 0, w[receiver].copy()
        elif state_r == "carry":
            k_r += 1
            w[receiver] = lerp_pose(start["r"], wrist_for_grip(place + [0, 0, 0.04], TOP_DOWN),
                                    min_jerk(carry)[k_r - 1])
            if k_r == carry:
                state_r = "done"
        # --- giver: decides from the receiver's *current* physical state ---
        if state_g == "carry":
            k_g += 1
            w[giver] = lerp_pose(start["g"], giver_at_exchange, min_jerk(giver_carry)[k_g - 1])
            if k_g == giver_carry:
                state_g = "hold"
        elif state_g == "hold":
            at_grasp = np.linalg.norm(b.wrist[-1][receiver, :3] - recv_grasp[:3]) < CAPTURE_RADIUS
            closed = b.aperture[-1][receiver] < CLOSED_BELOW
            captured_for = captured_for + 1 if (at_grasp and closed) else 0
            if captured_for * DT >= CAPTURE_DWELL:
                state_g, k_g = "open", 0
        elif state_g == "open":
            k_g += 1
            a[giver] = CLOSED + (OPEN - CLOSED) * k_g / opening
            if k_g == 1:
                release_row = b.now + 1
            if k_g == opening:
                state_g, k_g, start["g"] = "retreat", 0, w[giver].copy()
        elif state_g == "retreat":
            k_g += 1
            back = start["g"].copy()
            back[0] += side * 0.15
            w[giver] = lerp_pose(start["g"], back, min_jerk(retreat)[k_g - 1])
            if k_g == retreat:
                state_g = "done"
        b.tick(w, a)
        if state_g == "retreat" and k_g == 0 and transfer_row is None:
            b.transfer(giver, receiver)
            transfer_row = b.now
        if state_g == "done" and state_r == "done":
            break
        if t > 2000:
            raise RuntimeError("handover did not finish")
    b.hold(0.5)
    ep = b.episode([MOVABLE, FIXTURE], [colour_id("red"), colour_id(None)],
                   [category_id("microphone"), category_id(None)],
                   np.stack([OBJECT_SIZE, np.array([0.6, 0.3, 0.02])]), ["obj_0", "deck"],
                   {"task": "handover", "branch": branch, "giver": giver,
                    "release_row": release_row, "transfer_row": transfer_row,
                    "delay_at": delay_at})
    return ep


def family(giver: int = 0, **kw) -> dict[str, Episode]:
    """All branches with a shared ``cue`` annotation each (first row differing from
    nominal); nominal's cue is the earliest branch cue, so all members mask alike."""
    eps = {name: handover(name, giver=giver, **kw) for name in BRANCHES}
    nominal = eps["nominal"]
    cues = {}
    for name, ep in eps.items():
        n = min(ep.T, nominal.T)
        diff = np.flatnonzero(~np.all(np.isclose(
            np.concatenate([ep.wrist[:n].reshape(n, -1), ep.objects[:n].reshape(n, -1),
                            ep.aperture[:n]], 1),
            np.concatenate([nominal.wrist[:n].reshape(n, -1), nominal.objects[:n].reshape(n, -1),
                            nominal.aperture[:n]], 1), atol=1e-9), axis=1))
        cues[name] = int(diff[0]) if len(diff) else None
    first = min(c for c in cues.values() if c is not None)
    for name, ep in eps.items():
        ep.annotations["cue"] = cues[name] if cues[name] is not None else first
    return eps


def _holder_drift(ep: Episode, arm: int, rows: range, obj: int = 0) -> float:
    """Max change of the object's pose in the holder's frame over ``rows`` (m + rad)."""
    from .trajectory import relative

    rel = np.stack([relative(ep.wrist[t, arm], ep.objects[t, obj]) for t in rows])
    return float(np.abs(rel[:, :3] - rel[0, :3]).max()
                 + Q.angle(rel[:, 3:], np.broadcast_to(rel[0, 3:], rel[:, 3:].shape)).max())


def _grip_inside(ep: Episode, arm: int, t: int, obj: int = 0, margin: float = 1e-3) -> bool:
    from .trajectory import grip_point, relative

    local = relative(ep.objects[t, obj], np.concatenate([grip_point(ep.wrist[t, arm]),
                                                         Q.IDENTITY]))[:3]
    return bool(np.all(np.abs(local) <= ep.size[obj] / 2 + margin))


def handover_checks(eps: dict[str, Episode], history: int = 16, horizon: int = 8) -> dict:
    """The acceptance checks of the handover fixture."""
    from .schema import query_table, target_window

    nom, dly = eps["nominal"], eps["delayed"]
    cue = dly.annotations["cue"]
    giver = nom.annotations["giver"]
    receiver = 1 - giver
    same_inputs = same_targets = True
    anticipates = False
    for t in range(cue):
        a, b = query_table(nom, t, history).model_inputs(), query_table(dly, t, history).model_inputs()
        same_inputs &= all(np.array_equal(a[k], b[k]) for k in a)
        wa, va = target_window(nom, t, horizon, cue=cue)
        wb, vb = target_window(dly, t, horizon, cue=cue)
        same_targets &= np.array_equal(va, vb) and np.array_equal(wa[va], wb[vb])
        anticipates |= bool(np.any(t + 1 + np.flatnonzero(va) >= cue))
    extra = {name: (ep.annotations["release_row"] - nom.annotations["release_row"]) * DT
             for name, ep in eps.items()}
    rigid = {}
    for name, ep in eps.items():
        tr = ep.annotations["transfer_row"]
        rigid[name] = max(_holder_drift(ep, giver, range(0, tr)),
                          _holder_drift(ep, receiver, range(tr, ep.T)))
    compatible = {name: _grip_inside(ep, giver, ep.annotations["transfer_row"] - 1)
                  and _grip_inside(ep, receiver, ep.annotations["transfer_row"] - 1)
                  for name, ep in eps.items()}
    return {"cue": cue, "extra_hold_s": extra, "rigid_drift": rigid,
            "checks": {
                "inputs_identical_before_cue": bool(same_inputs),
                "targets_identical_before_cue": bool(same_targets),
                "no_window_anticipates_cue": not anticipates,
                "delayed_giver_holds_longer": extra["delayed"] > 0.2,
                "sham_does_not_change_release": abs(extra["sham"]) <= 2 * DT,
                "rigid_carry": max(rigid.values()) < 1e-6,
                "exchange_grasps_compatible": all(compatible.values()),
            }}
