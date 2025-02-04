# kinda like the torchmetrics _AbstractStatScores class but simpler
import torch
from torch import Tensor
from torchmetrics import Metric


class PanopticStatScore(Metric):
    # dont forget to call super().__init__(**kwargs) in subclasses.
    is_differentiable: bool | None = False
    higher_is_better: bool | None = True
    full_state_update: bool | None = False

    def _create_state(self, num_classes: int):
        """
        add tp, fp, fn and iou states. defaults will be zeros(num_classes),
        and reduce is sum.
        states will have a prefix p_, just in case to prevent conflicts
        with normal torchmetrics
        """
        self.add_state("p_tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("p_fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("p_fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("p_iou", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def _update_state(self, tp: Tensor, fp: Tensor, fn: Tensor, iou: Tensor):
        """
        add the tp, fp, fn, iou to respective states.
        """
        self.p_tp += tp
        self.p_fp += fp
        self.p_fn += fn
        self.p_iou += iou

    def _final_state(self):
        """
        for now just returns, tp, fp, fn, iou.
        """
        # but check torchmetrics _AbstractStateScores final_state for cat-ing lists
        return self.p_tp, self.p_fp, self.p_fn, self.p_iou
