from torch import Tensor

from .panoptic_quality_functional import (
    _panoptic_quality_compute,
    _panoptic_quality_update,
)
from .panoptic_stat_score import PanopticStatScore


# the idea behind the inheritence being you can eventually do RQ and SQ seperately if
# needed and take advantage of torchmetrics compute groups
class PanopticQuality(PanopticStatScore):
    """
    Taking point clouds as the use case, order of preds and targets has to be the same
    with respect to the points.
    take care that stuff classes instance labels should be 0. in both prediction and
    target tensors.
    Assumes also that semantics labels go from [0, num_classes) sequentially.

    :param num_classes: number of semantic classes. includes both stuff and thing
    :param average: the reduction to apply on class pq scores. "mean" gets mean (while
    ignoring ignore_id), "none" or None returns a tensor with same size as num_classes
    :param ignore_id: a single id that is ignored in target semantic labels. take note
    that this is not used when doing the instance part of the evaluation. this anyways
    doesn't make sense since there should be no "ignored" instances in a thing class,
    that should be predicted as an ignore semantic stuff instead.
    :param min_points: minimum number of points in a label to be considered for fn and
    fp calculation. tp matches match everything regardless of size. this is how its done
    in the original semantic-kitti panoptic eval which is reimplemented here. set this
    to a reasonable number based on your number of points per sample to avoid any adverse
    influence on metric calculations.
    :param minkowski: a flag to switch how to handle input batches. this is intended to
    be used with point clouds, and different samples in a batch might have different
    point counts. Minkowski needs this to be handled by adding a batch index column to
    coords and then vstacking everything. any other features and labels are vstacked
    without any additions. hence to properly compute metrics with minkowski batches,
    passing an additional batch_idx tensor is necessary which is the first column of
    your coords.
    :param threshold: the iou threshold above which a pred_id -> target_id
    match is considered true positive. there is no reason to change this from the
    default of 0.5.
    :param max_instances: this corresponds to 'offset' in the semantic kitti api
    original. this is assumed to be the maximum number of instances possible in a scene.
    there should be no reason to mess with this. this has been reduced from the original
    just in case to prevent overflow issues. consider that instance ids will be
    multipled by the batch_offset described below.
    :param batch_offset: the original semantic-kitti api cannot handle a batch with multiple
    samples at once as it is implemented. the batch offset here is used to multiply the
    pred and target instance labels per sample based on its batch_idx. this allows to
    have unique pred and target id matches per sample. there should be no reason to change
    this. the defaults are sane values unless you're trying to segment the world.
    """

    def __init__(
        self,
        num_classes: int,
        average: str | None = "mean",
        ignore_id: int | None = None,
        min_points: int = 50,
        minkowski=True,
        threshold: float = 0.5,
        max_instances: int = 2**20,
        batch_offset: int = 2**10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.average = average
        self.ignore_id = ignore_id
        self.min_points = min_points
        self.minkowski = minkowski
        self.threshold = threshold
        self.max_instances = max_instances
        self.batch_offset = batch_offset
        self._create_state(num_classes=self.num_classes)

    def update(
        self,
        pred_sem: Tensor,
        target_sem: Tensor,
        pred_inst: Tensor,
        target_inst: Tensor,
        batch_idx: Tensor | None = None,
    ) -> None:
        """
        updates the state using the preds and targets.
        if not minkowski, extra dimensions in labels get flattened into batch dimension.

        :param pred_sem: the predicted semantic ids, shape BxN or simply N if in
        minkowksi mode
        :param target_sem: the target semantic ids, shape BxN or N if in minkowksi mode
        :param pred_inst: the predicted instance ids, shape BxN or simply N if in
        minkowksi mode
        :param target_inst: the target instance ids, shape BxN or N if in minkowksi mode
        :param batch_idx: a tensor with the corresponding batch index for each element.
        this is relevant for minkowski mode, pass the first column of your coords there
        """
        if self.minkowski:
            assert (
                batch_idx is not None
            ), "Metric in minkowski mode needs batch_idx tensor"
        tp, fp, fn, iou = _panoptic_quality_update(
            pred_sem=pred_sem,
            target_sem=target_sem,
            pred_inst=pred_inst,
            target_inst=target_inst,
            num_classes=self.num_classes,
            ignore_sem_id=self.ignore_id,
            min_points=self.min_points,
            minkowski=self.minkowski,
            batch_idx=batch_idx,
            threshold=self.threshold,
            max_instances=self.max_instances,
            batch_offset=self.batch_offset,
        )
        self._update_state(tp=tp, fp=fp, fn=fn, iou=iou)

    def compute(self) -> Tensor:
        """computes the final statistic"""
        tp, fp, fn, iou = self._final_state()
        return _panoptic_quality_compute(
            tp=tp,
            fp=fp,
            fn=fn,
            iou=iou,
            average=self.average,
            num_classes=self.num_classes,
            ignore_sem_id=self.ignore_id,
        )
