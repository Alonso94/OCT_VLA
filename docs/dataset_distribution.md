# Portable dataset releases

A finalized LeRobot v3 directory contains parquet trajectories, metadata,
normalization statistics, task text, and encoded camera videos. Publish that
directory once; users then download the same release without regenerating data.

Publish RGB and object-conditioned datasets as separate repositories because
their feature schemas differ. Authenticate once with a Hugging Face write token
(`hf auth login`), then run from the policy environment:

```bash
PYTHONPATH=src /path/to/policy/python scripts/hub_dataset.py publish \
  outputs/lerobot/shelf_restock_atomic_25 \
  YOUR_ORG/oct-vla-shelf-restock-rgb --release v1.0.0 --private
```

After the object-token export is finalized, publish it in the same way:

```bash
PYTHONPATH=src /path/to/policy/python scripts/hub_dataset.py publish \
  /datasets/shelf_restock_atomic_25_object \
  YOUR_ORG/oct-vla-shelf-restock-object --release v1.0.0 --private
```

Each release uploads all data and videos, writes `dataset_manifest.json` with
SHA-256 hashes, and creates the supplied immutable Hub tag. Use a new tag for
corrected data, such as `v1.0.1`.

On any computer with Python and `huggingface_hub`, download a fixed release:

```bash
python scripts/hub_dataset.py download YOUR_ORG/oct-vla-shelf-restock-rgb \
  --release v1.0.0 --output /datasets/oct-vla-rgb-v1
```

Use the downloaded directory directly as `--dataset.root` in LeRobot training.

## Published releases

| Repository | Release | Episodes | Frames | Size |
| --- | --- | ---: | ---: | ---: |
| `3liyounes/oct-vla-shelf-restock-three_object-rgb` | `v1.0.0` | 123 | 17311 | ~137 MB |
| `3liyounes/oct-vla-shelf-restock-three_object-object` | `v1.0.0` | 123 | 17311 | ~138 MB |

Both are private. Each holds 90 train and 33 validation episodes in the episode
order described in [dataset_protocol.md](dataset_protocol.md#split-layout-inside-the-dataset),
so `split_manifest.json` travels with the data and `eval_split` needs no
restating at the download site. `octvla_episode_manifest.json` carries the source
seed and `episode.json` SHA-256 for every clip.

Verified after upload by downloading each release into a scratch directory and
re-checking all ten files against `dataset_manifest.json`: no mismatches. Doing
this round trip is worth the minute it takes -- it distinguishes a faithful
upload from one that merely reported success.

Note for anyone finetuning pi0.5 from these datasets: the policy's tokenizer
comes from the licence-gated `google/paligemma-3b-pt-224`, so a Hugging Face
account that has accepted that licence is required. The datasets themselves carry
no such restriction. See [setup_nhr_alex.md](setup_nhr_alex.md).
