"""In-training validation on fixed packs (dev, held-out, count)."""

from __future__ import annotations

import numpy as np

from oct_vla.bpfn.benchmark_tools.evaluator import Evaluator, decisions, evaluate, predict
from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN

HORIZONS = (1, 4, 8, 16)      # horizons reported separately (1-based)


class TrainingValidator(Evaluator):
    """Scores the current model on one pack: pose errors (overall, per horizon, the moving
    median and 90th percentile), paired both-correct, decision accuracy per family and, for
    colour rules, the right-colour rate. With ``shuffled``, also both-correct under shuffled
    supports (does the model still read the demonstration?).

    The scores are logged only. Training keeps one checkpoint (the latest) and never
    selects by a validation score: offline scores have not predicted closed-loop success here.
    """

    def __init__(self, items: list[dict], device: str = "cpu", shuffled: bool = False) -> None:
        super().__init__(items, device)
        self.shuffled = shuffled

    def run_validation(self, model: PoseWindowPFN) -> dict:
        windows = predict(model, self.items, self.s_rows, self.device)
        scores = evaluate(windows, self.items)
        per_h = scores.pop("trans_mm_per_h")
        for h in HORIZONS:
            if h <= len(per_h):
                scores[f"h{h}_mm"] = per_h[h - 1]
        scores.update(decisions(windows, self.items))
        if self.shuffled:
            rng = np.random.default_rng(self.seed)
            shuffled = self._condition_items("shuffled", rng)
            scores["shuffled_both_correct"] = self.score(model, shuffled)["both_correct"]
        return scores
