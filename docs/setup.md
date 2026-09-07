# Setup and external dependency discovery

The core and CLI require Python 3.10–3.12 and the standard library. Simulation
and training use separate environments. Commands never install dependencies,
download datasets/assets, clone repositories, or repair external environments.

## Reuse existing environments

From this repository root, invoke an existing interpreter directly, avoiding
stale console-script shebangs and old editable OCT-VLA imports:

```bash
PYTHONPATH=src /path/to/existing/environment/bin/python -B -m oct_vla doctor \
  --robotwin-root /path/to/existing/RoboTwin \
  --robotwin-python /path/to/simulator/environment/bin/python \
  --policy-python /path/to/policy/environment/bin/python \
  --data-root /path/to/existing/datasets
```

After an intentional package installation, the equivalent entrypoint is
`octvla doctor`. Only `doctor` is currently implemented. Do not install the old
project's combined dependencies into the simulator environment.

Copy `.env.example` to ignored `.env`, replace placeholders, then explicitly load
it with `doctor --env-file .env`. No file is auto-loaded. Supported dotenv syntax
is `KEY=value`, optional `export`, quoted values, blank lines, and comments.
Variable expansion, shell substitution, and arbitrary shell statements are not
supported or executed. Use literal paths, including for values referencing the
same installation.

The command also accepts `--config /path/to/paths.json` with a flat JSON object:

```json
{
  "robotwin_root": "/path/to/existing/RoboTwin",
  "robotwin_python": "/path/to/simulator/environment/bin/python",
  "policy_python": "/path/to/policy/environment/bin/python",
  "data_root": "/path/to/existing/datasets",
  "asset_root": "/path/to/existing/assets",
  "output_root": "../outputs"
}
```

JSON keeps discovery independent of third-party configuration libraries in both
environments. This is a path configuration, not the future experiment schema.
Keep machine-specific configuration files outside tracked paths (or use `.env`).

Resolution is CLI > process environment > explicit env file > JSON config >
derived defaults. Empty selected values and unknown file keys are errors.
Relative CLI/environment paths use the working directory; relative file values
use their configuration file's directory. `~` expands to the user's home.
Interpreter symlinks are preserved to retain virtual-environment semantics.

| CLI option / JSON key | Environment variable | Default |
| --- | --- | --- |
| `--robotwin-root` / `robotwin_root` | `ROBOTWIN_ROOT` | None |
| `--robotwin-python` / `robotwin_python` | `OCTVLA_ROBOTWIN_PYTHON` | `<robotwin_root>/.venv-robotwin/bin/python` |
| `--policy-python` / `policy_python` | `OCTVLA_POLICY_PYTHON` | None; explicitly select the learning environment |
| `--data-root` / `data_root` | `OCTVLA_DATA_ROOT` | None |
| `--asset-root` / `asset_root` | `OCTVLA_ASSET_ROOT` | `<robotwin_root>/assets` |
| `--output-root` / `output_root` | `OCTVLA_OUTPUT_ROOT` | None; doctor creates nothing |

## What doctor verifies

Doctor checks RoboTwin's `envs/` and camera/embodiment registries in either
`env_cfg/task_config/` or `task_config/`. It reports Git revision and tracked
modifications, directory presence, Python versions, package metadata, top-level
module resolution, stale plain-path `.pth` entries, and NVIDIA driver visibility
through `nvidia-smi`. It reports available cuRobo source as a candidate separately
from whether the interpreter resolves it.

Probes use the selected interpreters with `-I -B`: ambient `PYTHONPATH` and user
site-packages are ignored; bytecode writes are disabled. Installed environment
startup hooks still run as on normal Python startup. No simulator, PyTorch, or
policy module is imported by the probe. Each subprocess has a 10-second timeout.
`doctor` does not execute pip, uv, downloads, model loads, CUDA extension builds,
or filesystem write tests.

`FOUND` means discovered, not physically validated. `WARN` identifies caveats
such as tracked checkout changes, an unexpected Python minor version, or stale
editable paths. `FAIL` means a required discovery check failed. `INFO` gives
scope limitations. Exit codes are 0 with no failed discovery checks (warnings
may remain), 1 with failures, and 2 for invalid arguments/configuration. Use
`--json` for structured paths, checks, and exit code. JSON reports contain local
paths; keep saved machine reports outside version control.

Directory presence does not verify complete assets or valid datasets. Driver
visibility does not verify CUDA kernels, sufficient GPU memory, Vulkan rendering,
camera calibration, robot control, or policy inference. These remain integration
gates for subsequent commits. Untracked checkout files and native-library ABI
compatibility are not exhaustively inspected.

## Troubleshooting moved environments

If a module has a version but no source, inspect the reported missing editable
target. Moving a repository does not rewrite `.pth` paths or executable
shebangs. Invoke the environment's Python directly rather than its stale `pytest`
or `lerobot-train` entrypoint. The new source-tree invocation above handles OCT-VLA
resolution without editing the old installation.

Doctor deliberately does not add candidate cuRobo paths to the probe or mask the
broken installed mapping. A future simulator launcher must explicitly configure
the selected source path and verify native-extension imports; finding a source
directory alone is not sufficient. Do not repair or reinstall existing working
environments as a side effect of discovery.

`RoboTwinNativePort` (`robots/robotwin/native.py`) is that launcher: it inserts
`<robotwin_root>/envs/curobo/src` ahead of the stale editable install on
`sys.path` for its own process only, which resolves `import curobo` without
touching the simulator venv. Separately, `assets/embodiments/*/curobo*.yml`
(gitignored, generated by the original installation, not tracked by RoboTwin's
Git repository) can contain `urdf_path`/`collision_spheres` entries stamped as
absolute paths from wherever RoboTwin lived when those files were generated;
moving the checkout leaves those stale too, and cuRobo reads them as plain
file paths with no sys.path-style override available. This is not something
the native port repairs automatically; it is a one-time, explicitly reviewed
migration step, and not just path hygiene: these fields deliberately point at
RoboTwin's own `assets/embodiments/<name>/` URDF/collision copies rather than
cuRobo's bundled Franka model, so cuRobo's planning model and SAPIEN's
simulated robot stay pinned to the exact same kinematic/collision geometry by
construction — confirmed here by RoboTwin's `panda.urdf` carrying a
wrist-camera mount (`camera_joint`, `hand_to_camera_mount`) that cuRobo's
bundled URDF does not have. Regenerate with RoboTwin's own
`scripts/update_embodiment_config_path.py` (run from the RoboTwin root; it
substitutes `${ASSETS_PATH}` into every `*_tmp.yml` template and writes the
result over its `.yml` sibling) rather than hand-editing the stale paths or
repointing at cuRobo's bundled content directory. That script, not a hand
edit, is what resolved this on the inspected workspace; it touched all five
embodiments' generated cuRobo configs, all gitignored, none of them source.
Separately: the collision-spheres file here is byte-identical to cuRobo's
stock one, so the wrist-camera mount has no collision representation in
either — the planner cannot see it as an obstacle. That is an existing
RoboTwin characteristic, not something this path fix changes; it belongs in
held-object/collision-completeness notes for the oracle work, not here.

A GPU-driver failure may indicate host configuration or container/sandbox device
access. Compare the same read-only command on the intended execution host before
concluding that a driver installation is broken.

## Fresh installation on a new machine (manual only)

These steps are for a researcher without existing installations. They were not
executed during this rebuild. The snapshot below records the inspected old
workspace, not a fully validated clean-install lockfile for the new backend.

| Component | Observed reference |
| --- | --- |
| Native Python | 3.10.18 |
| RoboTwin | `cffb78057724c6899fbae7b71ec4cfb249fbc225`, with local modifications |
| cuRobo source | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` (distribution reports 0.0.0) |
| SAPIEN | 3.0.0b1 |
| Native PyTorch | 2.4.1 |
| Native NumPy requirement | 1.26.4 in the inspected simulator requirements |
| Policy Python | 3.12.3 |
| LeRobot | 0.6.2, Git `6adf51511b7625090eade8d82d9f61a1846ebe56` |
| Policy PyTorch | 2.11.0+cu128 |
| PEFT | 0.20.0 |

1. Provision an NVIDIA GPU host with driver, CUDA toolkit/compiler, Vulkan
   rendering support, and video encoding support suitable for the chosen native
   stack. Record driver/toolkit versions; the Torch build suffix alone does not
   establish the host toolkit version. Follow the
   [cuRobo installation guide](https://curobo.org/get_started/1_install_instructions.html)
   for its CUDA extension requirements.
2. Obtain RoboTwin in an external directory and select the reference revision
   above, including its pinned submodules. Follow the
   [RoboTwin installation and asset guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).
   Create a separate Python 3.10 environment and review that revision's native
   requirements/build scripts. The inspected workspace changed requirements and
   XPolicyLab files: those changes must be reviewed and captured before claiming
   exact reproduction. There is no verified patch bundle in this rebuild yet.
3. Install the selected SAPIEN and cuRobo native stack in that simulator environment.
   Keep the NumPy ABI consistent with its extensions. Obtain robot URDFs, meshes,
   calibration/configuration, and object assets through the selected RoboTwin
   release's documented asset procedure. Retain provenance and checksums.
4. Create a separate Python 3.12 policy environment. Use the
   [LeRobot installation guide](https://huggingface.co/docs/lerobot/installation)
   with the recorded source revision as a starting reference for π0.5, training,
   and PEFT dependencies. Preserve resolved versions; do not assume today's main
   branch reproduces this snapshot. No new π0.5 training adapter is implemented
   yet, so its final dependency lock and forward-pass validation remain future work.
5. Install this lightweight package intentionally into a chosen new development
   environment, from this checkout, with `python -m pip install -e '.[dev]'`.
   This installs packaging/dev tools, not RoboTwin or LeRobot. Alternatively,
   use `PYTHONPATH=src python -B -m oct_vla` without installation when dependencies
   already exist. Hatchling is required for building a wheel.
6. Obtain only the intended datasets. The
   [RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin)
   links its published trajectories and task-selective download procedures.
   Existing stack-blocks data is useful for integration checks, but it is not the
   planned 25 atomic shelf-restock demonstrations. Those require the future
   validated oracle and collection steps. Retain seeds, schema, sampling times,
   split definitions, and asset provenance with each dataset.
7. Set the paths above, run `doctor`, then run `python -B -m pytest -q` and
   `python -m ruff check .` from this checkout. After implementing the backend,
   validate a fixed-seed scene, cameras, actuation, planner collision handling,
   exact replay, and data conversion before launching training.

Fresh-install reproducibility remains limited by the old checkout's local changes
and unverified native runtime after its move. Commit 3 makes those gaps visible;
it does not certify a working simulator or shelf expert.
