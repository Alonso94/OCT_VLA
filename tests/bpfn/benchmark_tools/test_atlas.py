"""The atlas marker scan: a scorer check on analytic windows, and informative supports."""

import numpy as np

from oct_vla.bpfn.benchmark_tools import atlas
from oct_vla.bpfn.benchmark_tools import metrics


def test_expert_windows_score_perfectly_and_the_boundary_flips_with_the_support():
    rows = atlas.marker_scan(atlas.expert_predict)
    assert len(rows) == 2 * len(atlas.MARKERS)
    assert sum(r["tie"] for r in rows) == 2            # x = 0, once per support
    assert all(r["correct"] for r in rows if not r["tie"])
    near = {r["marker_x"]: r["analytic"] for r in rows if r["support"] == "nearest"}
    far = {r["marker_x"]: r["analytic"] for r in rows if r["support"] == "farthest"}
    flips = [m for m in near if near[m] is not None and near[m] != far[m]]
    assert len(flips) == len(atlas.MARKERS) - 1            # every non-tie cell flips


def test_hold_decides_nothing():
    rows = atlas.marker_scan(lambda items: [metrics.hold(it["query"], 16) for it in items])
    assert all(r["decided"] is None for r in rows)


def test_each_support_is_informative():
    """Nearest and farthest disagree on the support's own scene, so the support singles
    out its rule (a support where both rules pick the same object identifies nothing)."""
    for name, sup in atlas.supports().items():
        xs = sup.objects[0, :3, 0]
        marker = sup.objects[0, sup.track_ids.index("marker"), 0]
        d = np.abs(xs - marker)
        assert np.argmin(d) != np.argmax(d)
        moved = [j for j in range(3) if np.linalg.norm(sup.objects[-1, j, :3] - sup.objects[0, j, :3]) > 0.05]
        assert moved == [int(np.argmin(d) if name == "nearest" else np.argmax(d))]
