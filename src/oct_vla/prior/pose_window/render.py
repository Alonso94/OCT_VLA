"""Labelled figures for the first-slice fixtures (PNG + vector PDF), headless."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import COLOURS, DT, MOVABLE

_RGB = {"red": "#c0392b", "blue": "#2166ac", "green": "#1a9850", "yellow": "#d4a017",
        "unknown": "#7f7f7f"}
ARM_STYLE = {0: dict(color="#1b9e77", label="left wrist"), 1: dict(color="#7570b3", label="right wrist")}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out: Path) -> list[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out.with_suffix(".png"), out.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def _scene(ax, ep, title, moved=True, paths=True):
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((-0.42, -0.31), 0.84, 0.12, color="#eeeeee", zorder=0))
    ax.add_patch(Rectangle((-0.42, -0.12), 0.84, 0.12, color="#dddddd", zorder=0))
    ax.text(-0.41, -0.195, "lower shelf", ha="left", va="top", fontsize=6, color="#555555")
    ax.text(-0.41, -0.005, "upper shelf", ha="left", va="top", fontsize=6, color="#555555")
    for j, tid in enumerate(ep.track_ids):
        if ep.kind[j] != MOVABLE:
            continue
        c = _RGB[COLOURS[ep.colour[j]]]
        p0, p1 = ep.objects[0, j, :2], ep.objects[-1, j, :2]
        ax.add_patch(Rectangle(p0 - ep.size[j, :2] / 2, *ep.size[j, :2], color=c, zorder=2))
        ax.text(p0[0], p0[1] - 0.045, tid, ha="center", fontsize=6)
        if moved and np.linalg.norm(p1 - p0) > 0.05:
            ax.add_patch(Rectangle(p1 - ep.size[j, :2] / 2, *ep.size[j, :2], fill=False,
                                   edgecolor=c, lw=1.2, zorder=2))
            ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="->", color=c, lw=0.8), zorder=1)
    for arm in (0, 1) if paths else ():
        path = ep.wrist[:, arm, :2]
        ax.plot(path[:, 0], path[:, 1], lw=0.6, alpha=0.6, **ARM_STYLE[arm])
    ax.set_title(title, fontsize=8)
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.34, 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=7)
    ax.tick_params(labelsize=6)


def colour_fixture_figure(result: dict, out: Path) -> list[Path]:
    """Two supports and the shared query, with each support's expected first-pick set."""
    plt = _plt()
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for ax, name in zip(axes[:2], ("red", "blue")):
        ep = result["supports"][name]
        _scene(ax, ep, f"support: {ep.annotations['rule']} "
                       f"({ep.annotations['eligible_count']} of {ep.annotations['physical']} moved)")
    axes[0].set_ylabel("y (m)", fontsize=7)
    q = result["query"]
    _scene(axes[2], q, "query at t = 0 (same scene under both supports)", moved=False,
           paths=False)
    for name, style in (("red", "-"), ("blue", "--")):
        for tid in sorted(result["sets"][name]):
            j = q.track_ids.index(tid)
            axes[2].add_patch(Circle(q.objects[0, j, :2], 0.045, fill=False, ls=style, lw=1.0,
                                     edgecolor=_RGB[name], zorder=3))
    axes[2].plot([], [], color=_RGB["red"], ls="-", label="first picks | red support")
    axes[2].plot([], [], color=_RGB["blue"], ls="--", label="first picks | blue support")
    axes[2].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    ok = " · ".join(f"{k.replace('_', ' ')}: {'✓' if v else '✗'}" for k, v in result["checks"].items())
    fig.suptitle("Colour fixture: swapping the demonstration swaps the eligible set\n" + ok,
                 fontsize=8)
    paths = _save(fig, out)
    plt.close(fig)
    return paths


def handover_fixture_figure(eps: dict, checks: dict, out: Path) -> list[Path]:
    """Receiver progress, giver aperture and the extra hold, per branch."""
    from .trajectory import grip_point

    plt = _plt()
    giver = eps["nominal"].annotations["giver"]
    receiver = 1 - giver
    cue = checks["cue"]
    style = {"nominal": dict(color="#333333", ls="-"), "delayed": dict(color="#d95f02", ls="-"),
             "sham": dict(color="#666666", ls=":")}
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.2), sharex=True)
    for name, ep in eps.items():
        t = ep.time
        target = grip_point(eps["nominal"].wrist[eps["nominal"].annotations["transfer_row"], receiver])
        dist = np.linalg.norm(np.stack([grip_point(w) for w in ep.wrist[:, receiver]]) - target, axis=1)
        axes[0].plot(t, 100 * dist, label=name, **style[name])
        axes[1].plot(t, ep.aperture[:, giver], **style[name])
        axes[2].plot(t, ep.aperture[:, receiver], **style[name])
        r = ep.annotations["release_row"] * DT
        axes[1].axvline(r, lw=0.6, **{k: v for k, v in style[name].items() if k != "ls"}, ls="--")
    for ax in axes:
        ax.axvspan(0, cue * DT, color="#eeeeee", zorder=0)
        ax.axvline(cue * DT, color="#d95f02", lw=0.8)
        ax.tick_params(labelsize=7)
    axes[0].text(cue * DT, axes[0].get_ylim()[1] * 0.9, "  cue: delay observable", fontsize=7,
                 color="#d95f02", va="top")
    axes[0].text(0.02, axes[0].get_ylim()[1] * 0.9, "identical inputs,\ntargets masked past cue",
                 fontsize=6, va="top", color="#555555")
    axes[0].set_ylabel("receiver to grasp (cm)", fontsize=7)
    axes[1].set_ylabel(f"giver aperture\n({ARM_STYLE[giver]['label']})", fontsize=7)
    axes[2].set_ylabel(f"receiver aperture\n({ARM_STYLE[receiver]['label']})", fontsize=7)
    axes[2].set_xlabel("time (s), 15 Hz rows", fontsize=7)
    axes[0].legend(fontsize=7, loc="upper right")
    extra = checks["extra_hold_s"]["delayed"]
    ok = " · ".join(f"{k.replace('_', ' ')}: {'✓' if v else '✗'}" for k, v in checks["checks"].items())
    fig.suptitle(f"Delayed-receiver handover: the giver holds {extra:.2f} s longer "
                 f"(dashed: giver release)\n{ok}", fontsize=7)
    paths = _save(fig, out)
    plt.close(fig)
    return paths
