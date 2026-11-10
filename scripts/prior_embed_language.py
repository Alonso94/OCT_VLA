#!/usr/bin/env python
"""Embed every instruction the interaction prior can see, with a frozen text encoder.

The prior's instruction set is finite (`programs.ALL_INSTRUCTIONS`), so it is embedded
once and looked up by index during training and evaluation. No language model is trained.

    prior_embed_language.py --out $HPCVAULT/octvla-prior/language/minilm.pt

The encoder is `sentence-transformers/all-MiniLM-L6-v2` (384-d), loaded through transformers
from `$HF_HOME`. The file keeps its per-token states, which the prior attends to as keys. A
mean-pooled sentence vector is kept for diagnostics only: it blurs programs that differ in
one component (cosine 0.97 between such programs, the same as between paraphrases).
The file records the model name and the exact strings, and training refuses a table whose
strings differ from the current program set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from oct_vla.prior.programs import ALL_INSTRUCTIONS

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = AutoModel.from_pretrained(args.model).eval()
    with torch.no_grad():
        t = tok(list(ALL_INSTRUCTIONS), padding=True, truncation=True, return_tensors="pt")
        h = enc(**t).last_hidden_state
        m = t["attention_mask"][..., None].float()
        e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
        tokens = h * m
    if torch.isnan(e).any() or e.std(0).mean() < 1e-3:
        raise SystemExit(
            "degenerate embeddings: every instruction maps to (nearly) the same vector"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "instructions": list(ALL_INSTRUCTIONS),
            "embeddings": e,
            "tokens": tokens,
            "token_mask": t["attention_mask"].bool(),
        },
        args.out,
    )
    n, length, dim = len(ALL_INSTRUCTIONS), tokens.shape[1], e.shape[1]
    print(f"{n} instructions, {length} tokens x {dim} -> {args.out}")


if __name__ == "__main__":
    main()
