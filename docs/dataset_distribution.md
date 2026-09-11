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
