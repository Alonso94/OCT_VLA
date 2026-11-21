"""The pre-generated episode bank: deterministic units, lossless storage, and samplers
that keep each arm's item semantics."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oct_vla.bpfn.data import bank  # noqa: E402
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
    s = bank.BankSampler(episode_bank, _cfg("full", conflict_fraction=1.0))
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
    items = bank.BankSampler(episode_bank, cfg).sample_items(np.random.default_rng(1), 300)
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
    s = bank.BankSampler(episode_bank, _cfg("equal_count"))
    rng = np.random.default_rng(2)
    for (p, n), ids in s.pool.items():
        assert len(ids) >= 2
        for g in ids:
            assert all(s._support(rng, p, n, avoid=int(g)) != g for _ in range(10))


def test_train_from_bank_smoke(tmp_path, bank_dir, monkeypatch):
    from oct_vla.prior.pose_window import train as Tr

    orig = splits.programs_for
    monkeypatch.setattr(splits, "programs_for",
                        lambda arm="full": tuple(p for p in PROGRAMS if p in orig(arm)))
    cfg = Tr.train(tmp_path / "run", arm="full", steps=3, batch=4, width=16, layers=1,
                   workers=0, s_rows=32, device="cpu", eval_every=10, bank=bank_dir)
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
