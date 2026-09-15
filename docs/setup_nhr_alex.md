# Verified installation: NHR@FAU Alex

A record of an installation that was actually executed and validated on a GPU
node, unlike the manual outline in [setup.md](setup.md#fresh-installation-on-a-new-machine-manual-only).
Every version below was resolved by that install, not copied from an earlier
workspace. Cluster-specific details are marked; the version pins are not.

## Resolved versions

| Component | Version | Notes |
| --- | --- | --- |
| Simulator Python | 3.10.20 | uv-managed |
| RoboTwin | `cffb78057724c6899fbae7b71ec4cfb249fbc225` | clean tree, submodule `482367a` |
| cuRobo | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` | tag `v0.7.8` resolves to exactly this commit |
| SAPIEN | 3.0.0b1 | |
| Simulator PyTorch | 2.4.1+cu121 | from `scripts/requirements.txt` |
| NumPy (simulator) | 1.26.4 | pinned upstream; the compiled extensions need the 1.x ABI |
| mplib | 0.2.1 | patched, see below |
| setuptools (simulator) | **69.5.1** | SAPIEN 3.0.0b1 imports `pkg_resources`, removed in setuptools >= 81 |
| Policy Python | 3.12.13 | |
| LeRobot | 0.6.2 @ `6adf51511b7625090eade8d82d9f61a1846ebe56` | **not on PyPI** (latest published is 0.6.1); install from git |
| Policy PyTorch | 2.11.0+cu128 | |
| **transformers** | **5.5.4** | see the version trap below |
| PEFT | 0.20.0 | |

## Two failures that do not announce themselves

### transformers must satisfy LeRobot's pin

LeRobot 0.6.2 requires `transformers>=5.4.0,<5.6.0`. PEFT declares `transformers`
with **no upper bound**, so installing PEFT on its own pulls a much newer release
(5.17.0 here). With that version pi0.5's checkpoint fails to load: the saved keys
are `...vision_tower.vision_model.embeddings...` while the model expects
`...vision_tower.embeddings...`.

This is logged as `Warning: Could not load state dict`, **not** an error. Training
then proceeds from a partly randomly-initialised vision tower, produces a
plausible falling loss curve, and yields results that look like a finding about
the method rather than a broken environment. Pin transformers explicitly after
installing PEFT, and check the load is clean:

```bash
grep -c "Could not load state dict" <train log>   # must be 0
```

### pi0.5 needs a gated tokenizer

`lerobot/pi05_base` ships `config.json`, `model.safetensors` and processor
configs — **no tokenizer**. pi0.5 loads it from `google/paligemma-3b-pt-224`,
which is licence-gated. Reproducing this project therefore requires a Hugging
Face account that has accepted that licence, plus a token.

Cache it once so training jobs can run with `HF_HUB_OFFLINE=1`:

```bash
export HF_HOME=...            # the token is read from $HF_HOME/token
python -c "from transformers import AutoTokenizer; \
           AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')"
```

A token saved by `hf auth login` lands in `~/.cache/huggingface/token`; if
`HF_HOME` points elsewhere, copy it there or the library reports "not
authenticated" while the file plainly exists.

## Install outline

The two environments stay separate, as `setup.md` requires. RoboTwin's
`scripts/_install.sh` is a useful reference but was not run verbatim.

```bash
# Simulator (RoboTwin Python 3.10)
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git
git -C RoboTwin checkout cffb78057724c6899fbae7b71ec4cfb249fbc225
git -C RoboTwin submodule update --init --recursive
uv venv --python 3.10 RoboTwin/.venv-robotwin          # doctor's derived default
uv pip install -r RoboTwin/scripts/requirements.txt    # pins torch/numpy/sapien
uv pip install "setuptools==69.5.1" wheel              # before any --no-build-isolation build

module load cuda/12.8.1
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export TORCH_CUDA_ARCH_LIST="8.0;8.6"                  # A100 sm_80 + A40 sm_86
git -C RoboTwin/envs clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
uv pip install -e RoboTwin/envs/curobo --no-build-isolation
uv pip install "warp-lang==1.12.0"

# Policy (LeRobot Python 3.12)
uv venv --python 3.12 policy-venv
uv pip install "lerobot[dataset,training] @ git+https://github.com/huggingface/lerobot.git@6adf5151..."
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "peft==0.20.0"
uv pip install "transformers>=5.4.0,<5.6.0"            # AFTER peft; see above
```

`TORCH_CUDA_ARCH_LIST` is not optional: without it the cuRobo build targets only
the GPU visible during the build, and the extension fails on the other partition.
The build takes ~18 minutes on one A40.

Two upstream source patches, both from `scripts/_install.sh`:

* `mplib/planner.py` line 807 — drop `or collide` from the conditional, or the
  planner rejects valid plans.
* `sapien/wrapper/urdf_loader.py` — add `encoding="utf-8"` to the URDF/SRDF opens.

Then the assets, and **`python scripts/update_embodiment_config_path.py`**, which
is required rather than optional: it stamps absolute paths into the generated,
gitignored `assets/embodiments/*/curobo*.yml`. Verify the stamped
`urdf_path`/`collision_spheres` actually resolve before trusting a plan.

## Cluster-specific (Alex)

Filesystem split follows <https://doc.nhr.fau.de/data/filesystems/>: `$HOME`
(100 GB, backed up) holds source and the two virtualenvs; `$HPCVAULT` (1 TB,
backed up, mounted on compute nodes) holds RoboTwin's ~32 GB asset pack, the
Hugging Face cache, and training outputs. `$WORK` was avoided — its **group**
inode quota was near its hard limit, so a virtualenv there would have affected
every member of the group.

* **Vulkan works, despite the warning.** SAPIEN prints `Failed to find Vulkan ICD
  file ... may not work`, then renders correctly at ~385 fps on an A40: it falls
  back to its own bundled `nvidia_icd.json`, which points at the real driver.
  Pinning `VK_ICD_FILENAMES` at the system ICD measures identically, and forcing
  lavapipe (software) crashes outright — so there is no silent CPU-rendering
  path to guard against, and the sbatch files need no Vulkan environment.
* **`--mem` is rejected** for GPU jobs ("Do not specify --mem for GPU jobs!"), and
  every job must request at least one GPU. Both are reflected in `slurm/`.
* **`MaxArraySize` is 10000**, not the default 1001 that
  [dataset_protocol.md](dataset_protocol.md#reserved-seed-ranges) warns about.
  Reserved seeds stay below 1000 anyway, for portability.
* **torchcodec has no system FFmpeg to load**, so LeRobot silently falls back to
  PyAV — **204.8 ms/sample versus 21.6 ms/sample**, which makes the dataloader,
  not the GPU, set training throughput. PyAV's wheel bundles FFmpeg 7 under
  mangled filenames that `dlopen` cannot match; a directory of plain-SONAME
  symlinks onto those libraries, on `LD_LIBRARY_PATH`, restores torchcodec.

## Validating the result

`octvla doctor` reports discovery only. Run it on a **GPU node**, where
`nvidia-smi` exists, and follow it with checks it deliberately does not perform:

```bash
python -c "from curobo.wrap.reacher.motion_gen import MotionGenConfig"   # CUDA extension loads
python RoboTwin/scripts/test_render.py                                   # prints "Render Well"
python scripts/collect_shelf_restock.py --seeds 1000 ...                 # one real episode
```

A discarded seed here is a normal outcome, not an installation failure: the
oracle throws away a whole run the moment one transfer fails to place, and the
observed yield is roughly half.
