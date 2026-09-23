"""The final table must refuse unpinned rollouts and split tiers only on evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "collect_final_results", ROOT / "scripts/collect_final_results.py"
)
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)

TIER_IDS = {"seen": [1, 2, 3, 4], "heldout": [0, 6], "novel": [5]}


def write(tmp_path, name, transfers, *, tier="seen", profile="three_object", pinned=True,
          layout=None):
    ids = TIER_IDS[tier]
    if layout is None:
        layout = collect.COUNT_LAYOUT.get(profile, "independent")
    episodes = [
        {"seed": 800 + i, "profile": profile, "success": t >= 3,
         "transfers_completed": t, "objects_lifted": t, "layout": layout,
         **({"model_ids": [ids[0]]} if pinned else {})}
        for i, t in enumerate(transfers)
    ]
    report = {"episodes": episodes, **({"model_ids_requested": ids} if pinned else {})}
    (tmp_path / f"{name}.json").write_text(json.dumps(report))


def run(tmp_path, capsys, monkeypatch):
    out = tmp_path / "table.json"
    monkeypatch.setattr(sys, "argv", ["collect", str(tmp_path), "--output", str(out)])
    collect.main()
    return json.loads(out.read_text()), capsys.readouterr().out


def test_an_unpinned_rollout_is_rejected_not_counted(tmp_path, capsys, monkeypatch):
    write(tmp_path, "F-rgb-s1000-seen", [1, 0], pinned=False)
    with pytest.raises(SystemExit, match="no admissible"):
        run(tmp_path, capsys, monkeypatch)
    assert "cannot prove their tier" in capsys.readouterr().out


def test_a_file_whose_name_and_pin_disagree_is_rejected(tmp_path):
    write(tmp_path, "F-rgb-s1000-heldout", [1])  # pinned to the seen ids
    with pytest.raises(collect.Rejected, match="named 'heldout'"):
        collect.read(tmp_path / "F-rgb-s1000-heldout.json", "F")


def test_a_scene_outside_its_tier_is_rejected(tmp_path):
    write(tmp_path, "F-rgb-s1000-heldout", [1], tier="heldout")
    report = json.loads((tmp_path / "F-rgb-s1000-heldout.json").read_text())
    report["episodes"][0]["model_ids"] = [3]
    (tmp_path / "F-rgb-s1000-heldout.json").write_text(json.dumps(report))
    with pytest.raises(collect.Rejected, match="outside"):
        collect.read(tmp_path / "F-rgb-s1000-heldout.json", "F")


def test_tiers_are_pooled_when_no_shift_holds_across_seeds(tmp_path, capsys, monkeypatch):
    # held-out beats seen on one seed and trails on the other: no evidence.
    write(tmp_path, "F-rgb-s1000-seen", [1, 1])
    write(tmp_path, "F-rgb-s1000-heldout", [2, 2], tier="heldout")
    write(tmp_path, "F-rgb-s1001-seen", [1, 1])
    write(tmp_path, "F-rgb-s1001-heldout", [0, 0], tier="heldout")
    table, out = run(tmp_path, capsys, monkeypatch)
    assert table["identity_split"] is False
    assert table["identity"]["rgb/all"]["episodes_per_seed"] == 4
    assert "pooled" in out


def test_tiers_are_split_when_one_arm_drops_on_every_seed(tmp_path, capsys, monkeypatch):
    for seed in (1000, 1001, 1002):
        write(tmp_path, f"F-rgb-s{seed}-seen", [2, 1])
        write(tmp_path, f"F-rgb-s{seed}-heldout", [0, 1], tier="heldout")
    for seed in (1000, 1001, 1002):
        write(tmp_path, f"F-adaln-s{seed}-seen", [2, 1])
        write(tmp_path, f"F-adaln-s{seed}-heldout", [2, 1], tier="heldout")
    table, _ = run(tmp_path, capsys, monkeypatch)
    assert table["identity_split"] is True
    assert {"rgb/seen", "rgb/heldout"} <= set(table["identity"])
    # Tested per tier once split, never pooled across them.
    assert "rgb->kv_adaln/three_heldout" in table["paired"]
    assert "rgb->kv_adaln/three_object" not in table["paired"]


def test_object_count_and_pairing(tmp_path, capsys, monkeypatch):
    for seed in (1000, 1001):
        write(tmp_path, f"F-rgb-s{seed}-seen", [0, 0, 0])
        write(tmp_path, f"F-adaln-s{seed}-seen", [3, 3, 0])
        # The count job holds two profiles in one file.
        path = tmp_path / f"F-adaln-s{seed}-count.json"
        write(tmp_path, f"F-adaln-s{seed}-count", [2, 0], profile="two_object")
        two = json.loads(path.read_text())
        write(tmp_path, f"F-adaln-s{seed}-count", [1], profile="four_object")
        four = json.loads(path.read_text())
        two["episodes"] += four["episodes"]
        path.write_text(json.dumps(two))
    table, _ = run(tmp_path, capsys, monkeypatch)
    assert set(table["object_count"]) >= {
        "kv_adaln/two_object", "kv_adaln/three_object", "kv_adaln/four_object"
    }
    paired = table["paired"]["rgb->kv_adaln/three_object"]["success"]
    assert paired["pairs"] == 6
    assert paired["treatment_only_wins"] == 4 and paired["baseline_only_wins"] == 0


def test_the_budget_control_is_a_baseline_and_mechanisms_are_contrasted(
    tmp_path, capsys, monkeypatch
):
    for seed in (1000, 1001):
        for arm, transfers in (("rgb", [0, 0]), ("rgb_cont", [1, 0]),
                               ("entity", [1, 1]), ("adaln", [3, 1])):
            write(tmp_path, f"F-{arm}-s{seed}-seen", transfers)
    table, _ = run(tmp_path, capsys, monkeypatch)
    assert {"rgb->kv_adaln/three_object", "rgb_cont->kv_adaln/three_object",
            "kv->kv_adaln/three_object"} <= set(table["paired"])
    assert "rgb_cont->rgb/three_object" not in table["paired"]


def test_stage_survival_conditions_each_stage_on_the_one_before():
    rows = [{"lift": l, "transfers": t} for l, t in
            [(True, 3), (True, 2), (True, 1), (True, 0), (False, 0)]]
    stages = collect.stage_survival(rows, 3)
    assert stages == [("lift", 4, 5), ("T1|lift", 3, 4), ("T2|T1", 2, 3), ("T3|T2", 1, 2)]


def test_wilson_interval_brackets_the_rate_and_widens_with_few_trials():
    lo, hi = collect.wilson(2, 8)
    assert lo < 0.25 < hi
    assert (hi - lo) > (lambda b: b[1] - b[0])(collect.wilson(20, 80))
    assert collect.wilson(0, 0) == (0.0, 1.0)


def test_independent_two_object_layout_is_dropped_but_nested_is_kept(tmp_path, capsys,
                                                                     monkeypatch):
    for seed in (1000, 1001):
        write(tmp_path, f"F-rgb-s{seed}-seen", [1, 2])
        write(tmp_path, f"F-rgb-s{seed}-count", [1, 0], profile="two_object",
              layout="independent")
    rows, rejected = collect.load(tmp_path, "F")
    assert not [r for r in rows if r["profile"] == "two_object"]
    assert any("independent layout" in line for line in rejected)
    for seed in (1000, 1001):
        report = json.loads((tmp_path / f"F-rgb-s{seed}-count.json").read_text())
        for episode in report["episodes"]:
            episode["layout"] = "nested_in_3"
        (tmp_path / f"F-rgb-s{seed}-count.json").write_text(json.dumps(report))
    rows, rejected = collect.load(tmp_path, "F")
    assert len([r for r in rows if r["profile"] == "two_object"]) == 4 and not rejected


def test_stages_are_reported_per_seed_and_failure_modes_counted(tmp_path, capsys, monkeypatch):
    for seed, transfers in ((1000, [3, 1]), (1001, [0, 2])):
        write(tmp_path, f"F-rgb-s{seed}-seen", transfers)
    report = json.loads((tmp_path / "F-rgb-s1000-seen.json").read_text())
    report["episodes"][1]["events"] = {
        "obj_0": {"outcome": "placed"}, "obj_1": {"outcome": "dropped_before_shelf"},
        "obj_2": {"outcome": "never_lifted"},
    }
    (tmp_path / "F-rgb-s1000-seen.json").write_text(json.dumps(report))
    table, out = run(tmp_path, capsys, monkeypatch)
    stages = table["stages"]["rgb/all"]
    assert set(stages["per_seed"]) == {1000, 1001} or set(stages["per_seed"]) == {"1000", "1001"}
    assert stages["pooled"][0]["reached"] == 3 and stages["pooled"][0]["attempted"] == 4
    assert table["failure_modes"]["rgb"]["dropped_before_shelf"] == 1
    assert table["failure_modes"]["rgb"]["episodes"] == 1
    assert "FAILURE MODES" in out


def test_failure_order_matches_the_tracker():
    from oct_vla.tasks.shelf_restock.events import OUTCOMES

    assert set(collect.FAILURE_ORDER) == set(OUTCOMES) - {"placed", "never_lifted"}


def test_a_rerun_under_the_current_arm_name_supersedes_its_alias(tmp_path):
    write(tmp_path, "F-entity-s1000-count", [1, 1])
    write(tmp_path, "F-kv-s1000-count", [2, 2])
    write(tmp_path, "F-entity-s1001-count", [0, 1])
    rows, rejected = collect.load(tmp_path, "F")
    by_seed = {s: [r["transfers"] for r in rows if r["train_seed"] == s] for s in (1000, 1001)}
    assert by_seed == {1000: [2.0, 2.0], 1001: [0.0, 1.0]}
    assert any("pre-rename" in line for line in rejected)
