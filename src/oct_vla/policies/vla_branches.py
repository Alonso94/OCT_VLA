"""The encoder-side object arms on the VLA backbones, shared where hosts agree.

pi0.5 and SmolVLA both build their action-expert sequence in `embed_suffix`,
as (embeddings, pad masks, block-attention masks); GR00T builds its DiT
sequence inside the head, so it has its own hooks in `control_groot`. The
mechanisms mirror `control_act`'s (LPWM, arXiv:2603.04553):

* **AdaLN** (`SceneVector`): the pooled scene modulates every block of the
  action model. Exact identity at init.
* **In-context** (`InContextEntities`): entity tokens join the action model's
  sequence and are dropped from its output. Not identity at init.
"""

from __future__ import annotations

import torch
from torch import Tensor


def scene_inputs(control):
    """(tokens, mask) for this forward, or None when there is no scene."""
    inputs = control._inputs
    if inputs is None or inputs[0] is None or inputs[0].shape[-2] == 0:
        return None
    return inputs


def prepend_entities(control, embs: Tensor, pad_masks: Tensor, att_masks: Tensor):
    """Put the entity tokens in front of an `embed_suffix` sequence.

    In front, because both hosts read their actions back as the *last*
    `chunk_size` suffix positions. The entities open one attention block of
    their own: the block-causal mask (`make_att_2d_masks`) then lets actions
    attend to entities and entities to the prefix, but never entities to the
    noisy actions -- so the conditioning is the same at every denoising step.
    Padded slots are pad-masked out of attention as queries and keys.
    """
    incontext = getattr(control, "incontext", None)
    inputs = scene_inputs(control)
    if incontext is None or inputs is None:
        return embs, pad_masks, att_masks
    entities, padding = incontext.embed(*inputs, embs.shape[0], embs.dtype)
    opens = torch.zeros(padding.shape, dtype=att_masks.dtype, device=att_masks.device)
    opens[:, 0] = 1
    return (
        torch.cat([entities, embs], dim=1),
        torch.cat([~padding, pad_masks], dim=1),
        torch.cat([opens, att_masks], dim=1),
    )
