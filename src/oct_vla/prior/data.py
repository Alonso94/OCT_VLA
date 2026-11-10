"""Training stream and fixed evaluation sets of the interaction prior.

Training batches are generated on the fly, seeded by (seed, worker, step). Two runs with the
same seed and worker count therefore see the same data.

Evaluation sets are generated once and saved as compressed npz with a manifest: the
program split, the item config, the generator's git revision and the seed. Every
evaluation reads the same items. The chunk seed is a hash of (name, split, chunk), as in
GEMIL's `pfn/dataset.py`, so the set does not depend on how the generation was parallelised.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import programs as P
from .batches import ItemConfig, sample_batch


def chunk_seed(name: str, split: str, chunk: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{name}/{split}/{chunk}".encode()).digest()[:8], "little")


class Stream(torch.utils.data.IterableDataset):
    """Endless training batches; each worker draws from its own seeded generator."""

    def __init__(self, cfg: ItemConfig, batch_size: int, seed: int):
        self.cfg, self.batch_size, self.seed = cfg, batch_size, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])
        while True:
            yield sample_batch(rng, self.cfg, self.batch_size)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _chunk(args):
    name, split, c, cfg, size = args
    return sample_batch(np.random.default_rng(chunk_seed(name, split, c)), cfg, size)


def build_eval_set(
    out: Path,
    name: str,
    split: str,
    items: int,
    chunk: int = 100,
    workers: int = 8,
    cfg: ItemConfig | None = None,
) -> Path:
    """Generate `items` items of the program split `split` into `out/<split>.npz`."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}.npz"
    if path.exists():
        raise FileExistsError(f"{path} exists; evaluation sets are never overwritten")
    cfg = cfg or ItemConfig(programs=P.program_split(split))
    jobs = [(name, split, c, cfg, chunk) for c in range(-(-items // chunk))]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_chunk, jobs))
    # Re-index conflict partners: each chunk's `pair` is chunk-local.
    offset, fixed = 0, []
    for p in parts:
        p = dict(p)
        p["pair"] = np.where(p["pair"] >= 0, p["pair"] + offset, -1)
        offset += len(p["t_obj"])
        fixed.append(p)
    batch = {k: np.concatenate([p[k] for p in fixed]) for k in fixed[0]}
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    np.savez_compressed(tmp, **batch)
    os.replace(tmp, path)
    manifest = {
        "name": name,
        "split": split,
        "items": int(len(batch["t_obj"])),
        "chunk": chunk,
        "config": {
            f.name: [p.id for p in cfg.programs] if f.name == "programs" else getattr(cfg, f.name)
            for f in dataclasses.fields(cfg)
        },
        "git": _git_revision(),
    }
    (out / f"{split}.json").write_text(json.dumps(manifest, indent=1, default=str))
    return path


def load_eval_set(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_language(path: Path | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-token instruction states [n, L, D] and their mask [n, L] written by
    scripts/prior_embed_language.py, or None.

    Refuses a table whose strings differ from the current program set: indices would
    silently point at the wrong instructions."""
    if path is None:
        return None
    blob = torch.load(path, map_location="cpu")
    if blob["instructions"] != list(P.ALL_INSTRUCTIONS):
        raise ValueError(f"{path} embeds a different instruction set; regenerate it")
    if "tokens" not in blob:
        raise ValueError(f"{path} has no per-token states; regenerate it")
    return blob["tokens"].float(), blob["token_mask"].bool()
