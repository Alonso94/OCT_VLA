"""The learned future-keyframe source at evaluation: the frozen prior on the live scene.

The evaluation client calls it once per step, policy-side; the simulator server is not
involved. It is configured entirely from the training view's ``meta/info.json``
(``future_keyframe``, written by ``scripts/add_future_keyframe.py --keyframes pfn``):
- the same prior, language table, instruction and K;
- the same support pool (``meta/pfn_support.npz``, the view's training split).

So a rollout sees exactly the kind of input its policy trained on. Supports are drawn
per scene from that pool, seeded by (support seed, scene seed), so two arms evaluated
on a scene see the same demonstrations.

What it reads from an observation: the tracked scene, the EEF state and the support
geometry. It never reads ``observation.context`` (the task manager's target and phase);
``shelf_adapter.query`` takes no such argument.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import shelf_adapter as A


class PriorFutureSource:
    """Per step: PFN keyframe rows ``[3, 24]`` or latent cells ``[N_MAX + 1, W]``, their
    mask, and the prior's confidence (``shelf_adapter.confidence``)."""

    name = "pfn"

    def __init__(self, provenance: dict, dataset_root: Path, device: str = "cpu") -> None:
        from .data import load_language

        if provenance.get("source") != "pfn":
            raise ValueError(f"not a learned keyframe view: {provenance.get('source')!r}")
        self.kind = provenance["kind"]
        self.k = int(provenance["k"])
        self.support_seed = int(provenance["support_seed"])
        if provenance["instruction"] != A.INSTRUCTION or provenance["eef_to_grip"] != A.EEF_TO_GRIP:
            raise ValueError("the view was exported with a different adapter "
                             "(instruction or EEF offset); re-export it")
        self.device = device
        self.model = A.load_prior(provenance["prior"], device)
        if self.model.config != provenance["prior_config"]:
            raise ValueError(f"{provenance['prior']} is not the prior the view was exported with")
        self.lang = load_language(Path(provenance["lang"])) if provenance.get("lang") else None
        with np.load(Path(dataset_root) / provenance["support_pool"]) as pool:
            self.pool = [
                A.SupportDemo(pool["feats"][i], pool["types"][i], pool["rows"][i],
                              pool["event"][i], pool["gripper"][i], pool["flag"][i],
                              int(pool["seed"][i]))
                for i in range(len(pool["seed"]))]
        if len(self.pool) < self.k:
            raise ValueError(f"support pool of {len(self.pool)} cannot give K={self.k}")
        self._demos: list[A.SupportDemo] | None = None
        self._initial = None

    def reset(self, scene_seed: int, observation) -> list[int]:
        """Start an episode: keep its first frame and draw its support demos."""
        rng = np.random.default_rng([self.support_seed, scene_seed])
        chosen = sorted(int(i) for i in rng.choice(len(self.pool), self.k, replace=False))
        self._demos = [self.pool[i] for i in chosen]
        self._initial = (observation.scene, observation.eef)
        return [self.pool[i].seed for i in chosen]

    def __call__(self, observation) -> tuple[np.ndarray, np.ndarray, float]:
        if self._demos is None:
            raise RuntimeError("PriorFutureSource called before reset()")
        supports = getattr(observation, "supports", ())
        if not supports:
            raise ValueError("the learned keyframe needs the server's support geometry")
        q = A.query(*self._initial, observation.scene, observation.eef, supports)
        out = A.infer(self.model, A.pack(self._demos, [q]), self.lang, self.device,
                      latent=self.kind == "latent")
        conf = A.confidence(out, 0)
        if self.kind == "latent":
            return out["latent"][0].astype(np.float32), out["latent_ok"][0].astype(bool), conf
        rows, mask = A.keyframe_rows(out, 0, q)
        return rows, mask, conf
