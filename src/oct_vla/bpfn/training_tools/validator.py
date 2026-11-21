"""In-training validation on the fixed dev pack."""

from __future__ import annotations

from oct_vla.bpfn.benchmark_tools.evaluator import Evaluator
from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN


class TrainingValidator(Evaluator):
    """Scores the current model on the dev pack with its correct supports.

    The scores are logged only. Training keeps one checkpoint (the latest) and never
    selects by dev score: offline scores have not predicted closed-loop success here.
    """

    def run_validation(self, model: PoseWindowPFN) -> dict:
        return self.score(model)
