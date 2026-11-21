"""The pre-generated episode bank: deterministic units, lossless storage, and samplers
that keep each arm's item semantics."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oct_vla.bpfn.data import bank  # noqa: E402
from oct_vla.bpfn.training_tools.data import BankSampler  # noqa: E402
from oct_vla.bpfn.data import batches  # noqa: E402
from oct_vla.bpfn.data import splits  # noqa: E402

#: A partner-closed subset (each program's order flip is in it), with a handover program.
PROGRAMS = tuple(p for p in splits.programs_for("full")
                 if p.eligibility == "all" and p.order in ("left_to_right", "right_to_left")
                 and p.compaction == "none" and p.repeat == "until_empty"
                 and p.manipulation in ("carry_left", "handover"))
SPEC = bank.BankSpec(name="test", pool=2, conflict=1, handover=1, shards=2)


@pytest.fixture(scope="module")
def bank_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("bank")
    bank.build(root, SPEC, workers=2, programs=PROGRAMS, log=lambda *_: None)
    return root


@pytest.fixture(scope="module")
def episode_bank(bank_dir):
    return bank.Bank(bank_dir, log=lambda *_: None)


def _cfg(arm, **kw):
    return batches.ItemConfig(programs=tuple(p for p in PROGRAMS if p in splits.programs_for(arm)),
                         counts=splits.count_matrix_for(arm),
                         branch_fraction=0.0 if arm == "nominal_only" else 0.5, **kw)


def test_subset_is_partner_closed():
    assert {p.manipulation for p in PROGRAMS} == {"carry_left", "handover"}
    keys = {p.key for p in PROGRAMS}
    for p in PROGRAMS:
        assert batches.conflict_partner(p, np.random.default_rng(0))[0].key in keys


def test_units_do_not_depend_on_sharding():
    unit = bank.units(SPEC, PROGRAMS)[5]
    by = {p.key: p for p in PROGRAMS}
    a, _, _ = bank.generate_unit(SPEC, unit, by)
    b, _, _ = bank.generate_unit(bank.BankSpec(name="test", pool=2, conflict=1, handover=1,
                                           shards=7), unit, by)
    assert np.array_equal(a[0].wrist, b[0].wrist)
    c, _, _ = bank.generate_unit(bank.BankSpec(name="other"), unit, by)
    assert not np.array_equal(a[0].wrist, c[0].wrist)


def test_storage_round_trip():
    by = {p.key: p for p in PROGRAMS}
    eps = bank.generate_unit(SPEC, bank.units(SPEC, PROGRAMS)[2], by)[0]
    eps += bank.generate_unit(SPEC, bank.units(SPEC, PROGRAMS)[0], by)[0]
    packed = bank.pack_episodes(eps)
    for i, e in enumerate(eps):
        back = bank.unpack_episode(packed, i)
        for f in ("time", "wrist", "aperture", "objects", "size"):
            np.testing.assert_allclose(getattr(back, f), getattr(e, f), atol=1e-6)
        for f in ("kind", "colour", "category"):
            assert np.array_equal(getattr(back, f), getattr(e, f))
        assert back.track_ids == e.track_ids
        assert back.annotations["selected"] == e.annotations["selected"]


def test_bank_is_complete_and_resumable(bank_dir, episode_bank):
    assert episode_bank.episodes > 0 and not any(
        bank.shard_path(bank_dir, k).with_suffix("").name.endswith("tmp") for k in range(2))
    before = bank.shard_path(bank_dir, 0).stat().st_mtime_ns
    bank.build(bank_dir, SPEC, workers=1, programs=PROGRAMS, log=lambda *_: None)
    assert bank.shard_path(bank_dir, 0).stat().st_mtime_ns == before      # skipped
    kinds = np.bincount(episode_bank.records[:, 0], minlength=3)
    n_ho = sum(p.manipulation == "handover" for p in PROGRAMS)
    c = len(SPEC.counts)
    assert kinds.tolist() == [len(PROGRAMS) * c * SPEC.pool, len(PROGRAMS) * c * SPEC.conflict,
                              n_ho * c * SPEC.handover]


def test_refuses_another_bank_in_the_same_directory(bank_dir):
    with pytest.raises(SystemExit, match="different bank"):
        bank.build(bank_dir, bank.BankSpec(name="test", pool=3, conflict=1, handover=1, shards=2),
                 programs=PROGRAMS, log=lambda *_: None)


def test_refuses_an_incomplete_bank(tmp_path, bank_dir):
    (tmp_path / "bank.json").write_text((bank_dir / "bank.json").read_text())
    with pytest.raises(SystemExit, match="incomplete"):
        bank.Bank(tmp_path, log=lambda *_: None)


def test_conflict_pairs_share_inputs_and_split_on_the_support(episode_bank):
    s = BankSampler(episode_bank, _cfg("full", conflict_fraction=1.0))
    items = s.sample_items(np.random.default_rng(0), 40)
    assert all(i["meta"]["pair"] is not None for i in items)
    for a, b in zip(items[::2], items[1::2]):
        assert a["meta"]["pair"] == b["meta"]["pair"]
        assert a["meta"]["first_pick"] != b["meta"]["first_pick"]
        assert a["meta"]["program"] != b["meta"]["program"]
        assert a["meta"]["t"] < a["meta"]["diverge"] or a["meta"]["diverge"] == 0
        key = lambda q: np.sort(q.pose.reshape(len(q.time), -1), axis=1)  # noqa: E731
        np.testing.assert_array_equal(key(a["query"]), key(b["query"]))


@pytest.mark.parametrize("arm", ["full", "equal_count", "no_handover", "nominal_only"])
def test_arm_semantics(episode_bank, arm):
    cfg = _cfg(arm, conflict_fraction=0.3)
    items = BankSampler(episode_bank, cfg).sample_items(np.random.default_rng(1), 300)
    meta = [i["meta"] for i in items]
    allowed = {p.key for p in cfg.programs}
    assert {m["program"] for m in meta} <= allowed
    handover = [m for m in meta if m["program"].startswith("handover")]
    delayed = [m for m in meta if m["branch"] == "delayed"]
    if arm == "no_handover":
        assert not handover
    else:
        assert handover
    if arm == "nominal_only":
        assert not delayed
    elif arm != "no_handover":
        assert delayed and all(m["cue"] is not None for m in delayed)
    same = [m["eligible_support"] == m["eligible_query"] for m in meta]
    if arm == "equal_count":
        assert all(same)
    else:
        assert not all(same)
    batch = batches.pack(items[:16])
    assert batch["target"].shape == (16, batches.HORIZON, 2, 7)


def test_support_and_query_are_different_episodes(episode_bank):
    s = BankSampler(episode_bank, _cfg("equal_count"))
    rng = np.random.default_rng(2)
    for (p, n), ids in s.pool.items():
        assert len(ids) >= 2
        for g in ids:
            assert all(s._support(rng, p, n, avoid=int(g)) != g for _ in range(10))


def test_train_from_bank_smoke(tmp_path, bank_dir, monkeypatch):
    from oct_vla.bpfn.training_tools.trainer import PFNTrainer, TrainConfig

    orig = splits.programs_for
    monkeypatch.setattr(splits, "programs_for",
                        lambda arm="full": tuple(p for p in PROGRAMS if p in orig(arm)))
    cfg = PFNTrainer(TrainConfig(out=tmp_path / "run", arm="full", steps=3, batch=4, width=16,
                                 layers=1, workers=0, s_rows=32, device="cpu", eval_every=10,
                                 bank=bank_dir)).train()
    assert cfg["reuse"] == 1 and cfg["pair_extra"] == 0 and cfg["bank"]["episodes"] > 0
    assert (tmp_path / "run" / "DONE").exists()
    stored = json.loads((tmp_path / "run" / "config.json").read_text())
    assert stored["bank"]["path"] == str(bank_dir)


def test_synthetic_data_builds_bank_and_keeps_existing_packs(tmp_path):
    from oct_vla.bpfn.data.synthetic import SyntheticData

    data = SyntheticData(bank_root=tmp_path / "bank", bank_spec=SPEC, packs_root=tmp_path / "packs",
                         pack_names=("dev",), pack_items=4, s_rows=32, workers=1,
                         programs=PROGRAMS)
    data.generate_data()
    assert (tmp_path / "bank" / "bank.json").exists()
    assert all(bank.shard_path(tmp_path / "bank", k).exists() for k in range(SPEC.shards))
    pack = tmp_path / "packs" / "dev.pkl"
    before = pack.stat().st_mtime_ns
    data.bank_root = None
    data.generate_data()
    assert pack.stat().st_mtime_ns == before


# ---- grouped batches: one support encoded once, read by several queries ------------------

def test_grouped_forward_equals_one_support_per_item(episode_bank):
    from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN, to_torch

    torch.manual_seed(0)
    model = PoseWindowPFN(width=32, heads=4, layers=2).eval()
    # The output layer starts at zero (window = current wrist, whatever the support), which
    # would make any comparison of supports vacuous: randomise it.
    torch.nn.init.normal_(model.head[1].weight, std=0.5)
    groups = BankSampler(episode_bank, _cfg("full", conflict_fraction=0.5, s_rows=48)).sample_groups(
        np.random.default_rng(3), 6, 3)
    grouped = to_torch(batches.pack_groups(groups, s_rows=48))
    flat = to_torch(batches.pack(batches.ungroup(groups), s_rows=48))
    assert grouped["s_pose"].shape[0] == len(groups) < flat["s_pose"].shape[0]
    with torch.no_grad():
        a, b = model(grouped)["window"], model(flat)["window"]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    # A wrong mapping changes the answer (the index is not silently ignored).
    shifted = dict(grouped, support_index=(grouped["support_index"] + 1) % len(groups))
    with torch.no_grad():
        assert not torch.allclose(model(shifted)["window"], a, atol=1e-4)


def test_grouped_batch_refuses_a_bad_index(episode_bank):
    from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN, to_torch

    model = PoseWindowPFN(width=16, heads=2, layers=1).eval()
    groups = BankSampler(episode_bank, _cfg("full", s_rows=32)).sample_groups(np.random.default_rng(4), 2, 2)
    b = to_torch(batches.pack_groups(groups, s_rows=32))
    with pytest.raises(ValueError, match="does not map"):
        model(dict(b, support_index=b["support_index"] + len(groups)))
    no_index = {k: v for k, v in b.items() if k != "support_index"}
    with pytest.raises(ValueError, match="pass support_index"):
        model(no_index)


def test_conflict_groups_are_fully_paired(episode_bank):
    s = BankSampler(episode_bank, _cfg("full", conflict_fraction=1.0))
    rng = np.random.default_rng(5)
    for _ in range(5):
        ga, gb = s.group_pair(rng, 4)
        assert len(ga["items"]) == len(gb["items"]) == 4
        progs = {ga["items"][0]["meta"]["program"], gb["items"][0]["meta"]["program"]}
        assert len(progs) == 2
        key = lambda q: np.sort(q.pose.reshape(len(q.time), -1), axis=1)  # noqa: E731
        for x, y in zip(ga["items"], gb["items"]):
            assert x["meta"]["pair"] == y["meta"]["pair"] is not None
            assert x["meta"]["first_pick"] != y["meta"]["first_pick"]
            assert x["meta"]["t"] == y["meta"]["t"]
            assert x["meta"]["t"] < x["meta"]["diverge"] or x["meta"]["diverge"] == 0
            np.testing.assert_array_equal(key(x["query"]), key(y["query"]))
        # each group's queries all belong to its support's program
        for g in (ga, gb):
            assert len({it["meta"]["program"] for it in g["items"]}) == 1


@pytest.mark.parametrize("arm", ["full", "equal_count", "no_handover", "nominal_only"])
def test_group_arm_semantics(episode_bank, arm):
    groups = BankSampler(episode_bank, _cfg(arm, conflict_fraction=0.5)).sample_groups(
        np.random.default_rng(6), 60, 4)
    meta = [it["meta"] for g in groups for it in g["items"]]
    assert {m["program"] for m in meta} <= {p.key for p in _cfg(arm).programs}
    if arm == "no_handover":
        assert not any(m["program"].startswith("handover") for m in meta)
    if arm == "nominal_only":
        assert not any(m["branch"] == "delayed" for m in meta)
    same = [m["eligible_support"] == m["eligible_query"] for m in meta]
    assert all(same) if arm == "equal_count" else not all(same)
    share = np.mean([m["pair"] is not None for m in meta])
    assert 0.45 < share < 0.85          # 2f / (1 + f) = 0.67 at f = 0.5


def test_train_grouped_smoke(tmp_path, bank_dir, monkeypatch):
    from oct_vla.bpfn.training_tools.trainer import PFNTrainer, TrainConfig

    orig = splits.programs_for
    monkeypatch.setattr(splits, "programs_for",
                        lambda arm="full": tuple(p for p in PROGRAMS if p in orig(arm)))
    cfg = PFNTrainer(TrainConfig(out=tmp_path / "run", arm="full", steps=3, batch=8, width=16,
                                 layers=1, workers=0, s_rows=32, device="cpu", eval_every=10,
                                 bank=bank_dir, per_support=4)).train()
    assert cfg["per_support"] == 4 and (tmp_path / "run" / "DONE").exists()
    with pytest.raises(SystemExit, match="per_support"):
        PFNTrainer(TrainConfig(out=tmp_path / "run2", steps=1, batch=6, per_support=4,
                               bank=bank_dir, device="cpu"))


def _first_wrist_row(table):
    """Both wrists at the support's first row (row 0 is always kept), as float32 bytes."""
    rows = [table.pose[0, table.column("left")], table.pose[0, table.column("right")]]
    return np.stack(rows).astype(np.float32).tobytes()


def test_fresh_supports_are_new_episodes_and_keep_the_group_invariants(episode_bank):
    stored = set()
    for g in range(episode_bank.episodes):
        stored.add(episode_bank.episode(g).wrist[0].astype(np.float32).tobytes())
    cfg = _cfg("full", conflict_fraction=0.5, s_rows=48)

    from_bank = BankSampler(episode_bank, cfg).sample_groups(np.random.default_rng(8), 12, 3)
    assert all(_first_wrist_row(g["support"]) in stored for g in from_bank)    # the control

    fresh = BankSampler(episode_bank, cfg, fresh_supports=True).sample_groups(
        np.random.default_rng(8), 12, 3)
    assert not any(_first_wrist_row(g["support"]) in stored for g in fresh)
    assert all(g["items"] for g in fresh)
    assert any(it["meta"]["pair"] is not None for g in fresh for it in g["items"])

    # The bank-only sampler is unchanged by the option's existence.
    a = BankSampler(episode_bank, cfg).sample_items(np.random.default_rng(9), 20)
    b = BankSampler(episode_bank, cfg, fresh_supports=False).sample_items(np.random.default_rng(9), 20)
    pa, pb = batches.pack(a, s_rows=48), batches.pack(b, s_rows=48)
    assert all(np.array_equal(pa[k], pb[k]) for k in pa)


def test_training_logs_every_validation_pack(tmp_path, bank_dir, monkeypatch):
    from oct_vla.bpfn.training_tools.trainer import PFNTrainer, TrainConfig

    orig = splits.programs_for
    monkeypatch.setattr(splits, "programs_for",
                        lambda arm="full": tuple(p for p in PROGRAMS if p in orig(arm)))
    pack = BankSampler(bank.Bank(bank_dir, log=lambda *_: None),
                       _cfg("full", conflict_fraction=1.0, s_rows=32)).sample_items(
        np.random.default_rng(0), 8)
    cfg = TrainConfig(out=tmp_path / "run", steps=2, batch=8, width=16, layers=1, workers=0,
                      s_rows=32, device="cpu", eval_every=1, bank=bank_dir, per_support=4,
                      fresh_supports=True)
    PFNTrainer(cfg, {"dev": pack, "heldout": pack[:4]}).train()
    rows = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    scored = [r for r in rows if "dev" in r]
    assert len(scored) == 2 and all("heldout" in r and "both_correct" in r["dev"] for r in scored)
    assert json.loads((tmp_path / "run" / "config.json").read_text())["fresh_supports"] is True
