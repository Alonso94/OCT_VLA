# OCT-VLA

A research implementation for testing whether object-centric conditioning improves
few-shot VLA adaptation and compositional generalization over RGB-only adaptation.
The first experiment targets approximately 25 demonstrations of shelf restocking
with pretrained π0.5 and LoRA.

One demonstration transfers **one selected object from the lower shelf to the
upper shelf**, optionally compacts it toward a previously placed neighbor, and
retreats. A deterministic task manager will handle repeated transfers. Both arms
remain represented throughout the action sequence.

## Current status

Commits 1–3 establish packaging, canonical EEF states, 14-D Cartesian actions,
frame-labelled geometry, and read-only external dependency discovery through
`octvla doctor`. Robot control, perception, collection, training, and evaluation
are **planned, not implemented**. No simulator or policy dependency is imported
by the package.
See [architecture](docs/architecture.md) for boundaries, conventions, and gates.
See [coordinate frames](docs/coordinate_frames.md) and
[canonical actions](docs/canonical_action.md) for the implemented core API.

## Check this skeleton

Use an existing Python interpreter with pytest and Ruff installed, from the
repository root:

```bash
/path/to/existing/environment/bin/python -B -m pytest -q
/path/to/existing/environment/bin/python -m ruff check .
/path/to/existing/environment/bin/python -m ruff format --check .
```

Pytest resolves this repository's `src/` explicitly. The import smoke test also
runs without site-packages, so an old editable installation cannot satisfy it.
These commands do not install packages or download assets.

The dependency-free package targets Python 3.10–3.12 so shared contracts can run
in both the existing Python 3.10 simulator and Python 3.12 policy processes.
This does not imply that their dependency stacks can share one environment.
Wheel builds use Hatchling; development tools are declared in the `dev` extra.
See [setup](docs/setup.md) for manual fresh-install guidance, observed dependency
revisions, path configuration, and the limits of current validation.

## External resources

RoboTwin, SAPIEN, cuRobo, policy dependencies, assets, and datasets remain external.
Use [.env.example](.env.example) as a local path template; `.env` is ignored.
Load it explicitly with `octvla doctor --env-file .env`. Without installing the
package, use `PYTHONPATH=src /path/to/environment/bin/python -B -m oct_vla doctor`.
Normal commands must never silently install dependencies or download resources.

Existing environments moved with a repository can contain stale executable
shebangs and editable source paths. Doctor reports missing plain-path editable
targets and resolved dependency locations without repairing them. See
[setup and troubleshooting](docs/setup.md) before reusing an environment.

## Development workflow

Implement and validate one coherent step at a time. Each step stops for review;
pushing and proceeding require explicit approval. The old repository remains a
read-only source of evidence, and local machine paths stay out of tracked files.
