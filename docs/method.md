# Method: entity-set conditioning for pretrained robot policies

What the policy receives, the three ways it is injected, where each attaches in
every backbone, and how a conditioned run starts from an RGB one. The code is
`src/oct_vla/policies/conditioning/` (the method) and `src/oct_vla/policies/control_*/`
(where it attaches). For what each arm has measured, see `research_questions.md`.

## 1. The entity set

A scene is an **unordered set** of up to `N = 16` entities: the objects, both
grippers, and the support surfaces (the two shelf decks, or a task's table).
Padded rows are marked by the mask, not by a type. Each entity is one
17-column row, plus a boolean mask marking real rows
(`src/oct_vla/data/entity_tokens.py`, `conditioning/entity.py`).

| columns | content | units |
| --- | --- | --- |
| 0–2 | position | metres, workcell frame |
| 3–8 | rotation, 6D (two columns of the rotation matrix) | — |
| 9–11 | size | metres |
| 12 | gripper aperture (0 for non-grippers) | — |
| 13–16 | entity type, one-hot: movable object / left gripper / right gripper / support | — |

Properties the code guarantees:

- **No task leakage.** No target role, slot id, simulator handle, or object id.
  The policy is not told which object to move next.
- **A set, not a sequence.** Nothing depends on row order, so permuting the
  entities cannot change any output (tested).
- **Units kept apart.** The 13 continuous columns and the 4-column one-hot are
  projected separately, then summed, activated and normalised (`EntityEmbedding`).
- **Training-split statistics only.** Position and size are standardised with
  statistics fitted on the training split at export and carried in the policy
  config (`object_entity_normalizer`), so they travel with the checkpoint. The
  dataset stores raw values, and LeRobot's own normalisation of these tensors is
  disabled.
- **Padding is inert.** Padded rows are zeroed before embedding (even a NaN in a
  padded slot cannot reach a gradient), are masked out of every attention, and
  an empty scene contributes exactly zero.

*Planned:* a 32-column entity, 16 geometric plus 16 semantic, encoded
separately and concatenated. It will be a separate change with its own export
and rerun, so that its effect is not confounded with the cleanup.

## 2. The three arms

All three are **stage-2** additions to a trained RGB policy (§4), and they are
nested, so each isolates one addition:

| arm | adds | identity at init |
| --- | --- | --- |
| `kv` | the KV term in every hooked host attention layer | exact |
| `kv_adaln` | `kv` + scene AdaLN on every block | exact |
| `kv_tokens` | `kv` + entity tokens in the host's sequence | no (small, measured) |

Selected by one config field, `object_conditioning`, and by `CONDITIONING` in the
training scripts.

### 2.1 KV — ControlVLA's added attention term (`conditioning/kv.py`)

For a host attention layer with query `Q`, keys and values `K, V`, and output
projection `W_o, b_o`:

```
out = W_o [ softmax(Q Kᵀ/√d) V  +  softmax(Q K_zᵀ/√d) V_z ] + b_o
      K_z = W_k·E(z),  V_z = W_v·E(z),   W_k, W_v, their biases initialised to 0
```

`E(z)` is the entity embedding. The branch computes only the added term, using
the host's **own** projected query (after its positional transform) and its own
output weight, and it adds no bias (the host applies `b_o` once). It also uses
the host's attention dropout, so both terms are regularised alike.

This is the mechanism of ControlVLA (arXiv:2506.16211; official
`kvcontrol_transformer.py`): two independently normalised softmaxes, a shared
query and output projection, and zero-initialised control K/V. It differs from
ControlVLA in three deliberate ways:
- the control set is an explicit entity set rather than visual object features;
- padding is masked;
- there is no positional code over entities, because a set has no order.

### 2.2 AdaLN — scene modulation (`conditioning/adaln.py`)

The entity set is pooled by attention with one learned query (masked, and
permutation-invariant) into a scene vector `c`. `c` drives a zero-initialised
modulation of every block. This is *inspired by* LPWM's context conditioning
(arXiv:2603.04553, official `ctx_mode="adaln"`, after DiT). It is not a port:
LPWM conditions a particle dynamics model with per-particle context, while here
one scene summary conditions a pretrained policy.

It takes two forms, depending on whether the host already has adaptive norms:

- **`SceneAdaLN`: the host has none.** This is ACT, which is post-norm. Every
  sublayer becomes `x = LN(x + (1+α)·f(x))·(1+γ) + β`, with α, β and γ from
  `c`. DiT's `α·f(x)` with α = 0 would delete the pretrained sublayer, so every
  quantity is a delta around identity.
- **`SceneVector`: the host modulates its blocks from a condition vector.** The
  scene is *added* to that vector, so it reaches every block through the host's
  own pretrained modulation layers. For SmolVLA, which has no such vector, it
  instead drives a zero-initialised FiLM, `out·(1+γ)+β`, on every expert RMSNorm.

### 2.3 Tokens — entities in the host's sequence (`conditioning/tokens.py`)

Each entity becomes a host-width token (its embedding, projected, plus a learned
type code). The tokens join the host's sequence, the host attends over them, and
they are removed again before anything reads the host's output. This is inspired
by LPWM's `ctx_mode="token"`, with every entity rather than one context vector
as a token.

**It is not identity at init:** the new keys enter the host's softmax. A
learned logit gate `b`, initialised to −4, holds the entities to about e⁻⁴ of a
native token's attention mass. That is small, but the gradient into `b` stays
alive; at −∞ the arm is exactly the stage-1 policy, which the tests check.

Every host gets the same gate (since 2026-09-24):
- **ACT:** a float key-padding mask on the encoder (`EntityTokens.extend`).
- **pi0.5:** added to the additive 4-D mask, on action queries × real entity
  keys only (`gate_suffix_mask`), in a wrapper around `paligemma_with_expert.forward`.
- **SmolVLA:** the same bias, in its self-attention layers, through an eager
  attention that adds it before the boolean mask (`hosts._biased_eager`); its
  cross-attention layers attend the prefix only, so hold no entity keys.
- **GR00T: unsupported.** Its DiT takes no attention mask, so the tokens enter
  ungated. On the smoke test's barely trained stage 1 the step-0 drift was
  0.16 %; on the real stage-1 checkpoints it is 18–100 %, and the init check
  refuses the arm. `submit_vla_experiments.sh` refuses `kv_tokens` on GR00T,
  and it is reported as unsupported there.

The VLA hosts also number RoPE positions by a cumulative sum over the pad mask,
so prepended entities used to push every action N positions along.
`realign_suffix_positions` puts each action back where stage 1 had it and
gives the entities the first action's position. Before these two fixes the
pi0.5 action chunk moved 137 % at step 0 (loss 0.250 → 0.278) and SmolVLA's
63 %. After them, both are stage 1 at a gate of −∞ (pi0.5 to 1e-7, float
rounding; SmolVLA exactly) and move 0.36 % (pi0.5) and 3.1 % (SmolVLA) at
−4. The init check now fails the arm above 10 %.

## 3. Where each arm attaches

| backbone | `kv` | `kv_adaln` | `kv_tokens` |
| --- | --- | --- | --- |
| ACT (`control_act`) | every decoder cross-attention (LeRobot ACT has **one** decoder layer) | `SceneAdaLN` at all 11 main encoder/decoder sublayers; never the VAE encoder | appended to the encoder sequence behind the gate; stripped before the decoder |
| pi0.5 (`control_pi05`) | every action-expert attention layer, through Gemma's own query (`hosts.install_pi_kv`) | added to `adarms_cond`, the time vector of every expert layer's adaptive RMSNorm | prepended to the expert suffix as their own attention block, behind the gate, actions at their stage-1 positions |
| SmolVLA (`control_smolvla`) | every expert attention layer (`hosts.install_smol_kv`) | FiLM on every expert RMSNorm | prepended to the expert suffix, behind the gate in self-attention layers, actions at their stage-1 positions |
| GR00T N1.7 (`control_groot`) | every DiT attention layer, before its output projection (`hosts.install_diffusers_kv`) | added to the DiT timestep embedding | **unsupported** (ungated; refused by the init check) |
| VLA-JEPA (`control_vla_jepa`) | every DiT attention layer (`hosts.install_diffusers_kv`) | — | — |

Why the encoder-side arms exist: on ACT, `kv` alone is a single seam, and the
four encoder layers that fuse images with proprioception never see the objects.

For the pi0.5 and SmolVLA token arms, the suffix block mask
(`make_att_2d_masks`) lets:
- actions attend to entities;
- entities attend to the prefix;
- but never entities attend to the noisy actions.

So the conditioning is the same at every denoising step. Actions are still read
back as the last `chunk_size` positions.

GR00T's DiT takes no attention mask, so only real entities are inserted, which
requires every row of a batch to hold the same number (checked). Flow-matching
heads that repeat their batch for noise samples (GR00T, VLA-JEPA) get the
entities tiled the same way (`entity.repeat_to`).

All of it mounts by forward hooks and attribute assignment. No host module is
wrapped, so every pretrained state-dict key is unchanged, and the conditioning
lives in one subtree (`…object_conditioning.*`), which is the only thing stage 2
adds.

## 4. The two-stage recipe

Following ControlVLA, whose "without pretraining" ablation fails:

1. **Stage 1:** the stock policy on RGB and proprioception (`rgb`).
2. **Stage 2:** the conditioned policy, loaded from that checkpoint, with the new
   branches initialised as in §2. For `kv` and `kv_adaln`, it is numerically the
   stage-1 policy at step 0; this is tested bit-for-bit on ACT.

Controls:
- **`rgb_cont`:** stage 1 continued for as many steps as stage 2 adds. Stage 2
  trains *on top of* stage 1, so against `rgb` alone a gain could be extra
  training; `rgb_cont` is the fair baseline.
- **`scratch_kv`:** `kv` trained with no stage 1, ControlVLA's failing ablation,
  kept as one cell.

How stage 2 loads stage 1:

| backbone | stage-1 checkpoint | stage 2 starts from |
| --- | --- | --- |
| ACT | full weights | `last/pretrained_model` directly |
| pi0.5, SmolVLA | a LoRA adapter over the public base | a *merged* checkpoint (`scripts/merge_stage1_adapter.py`), plus a fresh adapter |
| GR00T | slim: trained head only, frozen backbone rebuilt from the base | `last/pretrained_model` directly |

Three guards, because every loader here fails quietly in its own way:

- **Verified loads** (`policies/stage_loading.py`). Every tensor in the file is
  compared with the model it was loaded into. Only the conditioning subtree, or a
  slim checkpoint's declared rebuilt prefixes, may be absent.
  - pi0.5's loader returns an unloaded model on failure, printing a warning.
  - LeRobot's generic loader drops keys silently.
  - GR00T's strict loader refused correct stage-2 loads.
- **The merge refuses** a zero adapter, a merge that changes the predicted
  chunk, and a saved checkpoint that reloads differently.
- **The init check** (`scripts/check_stage2_init.py`, `INIT_CHECK=1`) builds the
  run exactly as training would, and proves two things before training starts:
  - stage 2 predicts what stage 1 does;
  - gradient reaches the arm's branch.

## 5. Code map

| file | holds |
| --- | --- |
| `policies/conditioning/entity.py` | entity contract, `EntityEmbedding`, padding/tiling helpers |
| `policies/conditioning/kv.py` | `KVAttention`, `packed_mha_query` |
| `policies/conditioning/adaln.py` | `SceneAdaLN`, `SceneVector` |
| `policies/conditioning/tokens.py` | `EntityTokens`, `prepend_entities` |
| `policies/conditioning/hosts.py` | KV hooks for Gemma, SmolVLA and diffusers attention |
| `policies/object_conditioning.py` | shared config fields (and legacy-checkpoint migration), the mounted module, the policy mixin |
| `policies/stage_loading.py` | verified loads, slim checkpoints |
| `policies/control_*/modeling_*.py` | where each arm attaches in that backbone |
| `policies/masked_pi05/`, `policies/slim_groot/` | the RGB arms' pi0.5 (padding-masked loss) and GR00T (slim checkpoint) |

Checkpoints written before the 2026-09-23 cleanup still load: their removed
config fields are accepted only at the one value the code still implements.
Mechanisms that were removed (`controlvla`, `pooled`, 15-D object tokens,
role-stripped and privileged arms) live at git tag `stageA-2026-09-23`.
