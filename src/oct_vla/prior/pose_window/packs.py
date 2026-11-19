"""Fixed evaluation packs (plan 0.4/0.5): generated once from hashed split streams.

A pack is ``<out>/<name>.pkl`` (the items: tables, targets, evaluator meta) plus
``<name>.json`` (manifest: split, counts, programs, item count, schema). Packs are never
overwritten. Names:

- ``dev`` / ``test``: training programs, the ``full`` count matrix, half conflict pairs;
- ``heldout_handover_colour``: the sealed composition, support 3, query 2-4, pairs only;
- ``count_eval``: training programs, support exactly 3, query 2/3/4 (``eval3``);
- ``sealed_count``: support 3, query 5 eligible (extrapolation; seal until selection).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from . import batches as Bt
from . import generator as Gn
from . import splits as Sp
from .schema import SCHEMA_ID

PACKS = {
    "dev": dict(split="dev", programs="train", counts=Sp.CountMatrix("full"), conflict=0.5),
    "test": dict(split="test", programs="train", counts=Sp.CountMatrix("full"), conflict=0.5),
    "heldout_handover_colour": dict(split="heldout_handover_colour", programs="heldout",
                                    counts=Sp.CountMatrix("eval3"), conflict=1.0),
    "count_eval": dict(split="test", programs="train", counts=Sp.CountMatrix("eval3"), conflict=0.5),
    "sealed_count": dict(split="sealed_count", programs="train",
                         counts=Sp.CountMatrix("eval3", counts=(Sp.SEALED_COUNT,)), conflict=0.0),
}


def build(name: str, out: Path, items: int, s_rows: int = Bt.S_ROWS) -> Path:
    spec = PACKS[name]
    path = Path(out) / f"{name}.pkl"
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    programs = Sp.heldout_programs() if spec["programs"] == "heldout" else Sp.programs_for("full")
    cfg = Bt.ItemConfig(programs=programs, counts=spec["counts"],
                        conflict_fraction=spec["conflict"], s_rows=s_rows)
    out_items, k = [], 0
    while len(out_items) < items:
        rng = Sp.item_rng(spec["split"], k)
        k += 1
        try:
            out_items.extend(Bt.sample_pair(rng, cfg) if rng.random() < cfg.conflict_fraction
                             else [Bt.sample_item(rng, cfg)])
        except Gn.Infeasible:
            continue
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(out_items, f, protocol=5)
    manifest = {"name": name, "schema": SCHEMA_ID, "split": spec["split"], "items": len(out_items),
                "draws": k, "counts": spec["counts"].kind, "programs": len(programs),
                "pairs": sum(i["meta"]["pair"] is not None for i in out_items) // 2,
                "s_rows": s_rows, "horizon": Bt.HORIZON, "history": Bt.HISTORY}
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n")
    return path


def load(path: Path) -> list[dict]:
    with Path(path).open("rb") as f:
        return pickle.load(f)
