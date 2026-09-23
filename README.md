# OCT-VLA

Does conditioning a pretrained robot policy on an explicit **object/entity set**
help over RGB alone — and does *where* the objects enter the policy change what
it generalises to?

The testbed is a dual-arm Franka Panda **shelf-restock** task in RoboTwin/SAPIEN
(move every object from the lower shelf to the upper one). It has a controlled
object-identity holdout and 2/3/4-object scenes. The policies are ACT, pi0.5,
SmolVLA, GR00T N1.7 and VLA-JEPA, through LeRobot 0.6.2. Every conditioned
policy is trained in two stages: an RGB policy first, then the same policy with
object conditioning added at zero-initialisation, as in ControlVLA.

## Where things stand

The living scientific state is [docs/research_questions.md](docs/research_questions.md):
the four questions, what has been measured, and what is running.

In short, on ACT (3 training seeds, pinned identity tiers):
- object conditioning helps, most of all on object sizes never seen in training;
- in-context entity tokens are best on seen objects, and scene AdaLN on held-out
  ones;
- no arm generalises to 2 or 4 objects.

All of this is preliminary until the training-budget control (`rgb_cont`)
finishes. The VLA pipeline is built and smoke-tested but has not been trained.

## How it works

```
simulator / perception ──► ObjectScene + robot state ──► entity set [N,17] + mask
                                                                │
                           kv        ControlVLA's added K/V attention term
                           kv_adaln  + a pooled scene vector modulating every block
                           kv_tokens + the entities as tokens in the host's sequence
                                                                │
                              stage-1 RGB policy (ACT / pi0.5 / SmolVLA / GR00T / VLA-JEPA)
                                                                │
                                                             actions
```

- [docs/method.md](docs/method.md): the entity representation, the three arms,
  where each attaches in every backbone, and how stage 2 loads stage 1.
- [docs/data_protocol.md](docs/data_protocol.md): collection, splits and
  holdouts, control-space views, the entity contract, and the evaluation
  protocol.
- [docs/reproducibility.md](docs/reproducibility.md): environments, the cluster,
  the storage budget, and the pipeline end to end.

## Code map

| path | holds |
| --- | --- |
| `src/oct_vla/policies/conditioning/` | the method: entity embedding, KV, AdaLN, tokens, host hooks |
| `src/oct_vla/policies/control_*/` | one adapter per backbone: where each arm attaches |
| `src/oct_vla/policies/object_conditioning.py`, `stage_loading.py` | shared config and wiring; verified checkpoint loads |
| `src/oct_vla/data/` | episodes, entity tokens, LeRobot export, control-space views |
| `src/oct_vla/tasks/shelf_restock/` | the task: spec, oracle, success, RoboTwin environment |
| `src/oct_vla/serve/` | the simulator ↔ policy bridge used for every rollout |
| `scripts/`, `slurm/` | collection, export, training, evaluation, results |

## Running the tests

```bash
source ~/octvla/env-leftmost.sh
PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" -m pytest tests/ -q
```

`$OCTVLA_POLICY_PYTHON` is the only interpreter with torch and LeRobot. Under a
bare `python`, every policy test is skipped.

The simulator runs in a separate interpreter (`$OCTVLA_ROBOTWIN_PYTHON`), which
is why rollouts run as a server and a client.

Mechanisms removed in the 2026-09-23 cleanup remain at git tag
`stageA-2026-09-23`: the `controlvla` and `pooled` injection modes, the 15-D
object tokens, and the privileged arm.
