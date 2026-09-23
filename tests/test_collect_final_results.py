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


def write(tmp_path, name, transfers, *, tier="seen", profile="three_object", pinned=True):
    ids = TIER_IDS[tier]
    episodes = [
        {"seed": 800 + i, "profile": profile, "success": t >= 3,
         "transfers_completed": t, "objects_lifted": t,
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
