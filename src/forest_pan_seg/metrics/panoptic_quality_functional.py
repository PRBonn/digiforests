# this is essentially reworking semantic-kitti-api panoptic eval for use with torchemetrics.
# there are some further notes at the end of the file
# The MIT License
#
# Copyright (c) 2019, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import torch
from torch import Tensor

from digiforests_dataloader.utils.logging import logger


def _format_input(
    pred_sem: Tensor,
    target_sem: Tensor,
    pred_inst: Tensor,
    target_inst: Tensor,
    num_classes: int,
    ignore_sem_id: int | None = None,
    minkowski: bool = True,
    batch_idx: Tensor | None = None,
    batch_offset: int = 2**10,
):
    """
    if minkowski, handles adding the batch_offset to instance ids.
    if not minkowski, handles flattening and adding the batch_offset to instance ids
    all shapes should be similar. undefined behaviour otherwise
    """
    if minkowski:
        assert batch_idx is not None, "batch idx is needed"
        batch_idx = batch_idx.flatten().int()
    else:
        if pred_sem.ndim == 1:
            logger.warning(
                "pq metric not in minkowski mode, but input is not batched. "
                "this could be a fuckup if unintended. "
                "better to reshape into a batch dimension to be sure"
            )
            batch_idx = torch.zeros_like(pred_sem).int()
        elif pred_sem.ndim == 2:
            # create a batch_idx like with minkowski mode so we can repeat the
            # multiplication by batch_offset
            B, N = pred_sem.shape
            batch_idx = torch.arange(B).repeat(N, 1).int()
            batch_idx = batch_idx.T.flatten().to(pred_sem.device)
        else:
            raise Exception(f"shape is improper pred_sem - {pred_sem.shape}")

    pred_sem = pred_sem.flatten().int()
    target_sem = target_sem.flatten().int()
    pred_inst = pred_inst.flatten().int()
    target_inst = target_inst.flatten().int()
    assert (
        batch_idx.shape == target_inst.shape
    ), f"batch idx shape {batch_idx.shape} is not same as instance {target_inst.shape}"

    # make sure instances are not zeros
    pred_inst = pred_inst + pred_inst.min() + 1
    target_inst = target_inst + target_inst.min() + 1

    # allow handling of batches by update logic
    pred_inst = pred_inst + (batch_idx * batch_offset)
    target_inst = target_inst + (batch_idx * batch_offset)

    if ignore_sem_id is not None:
        assert ignore_sem_id in range(
            0, num_classes
        ), f"ignore sem id {ignore_sem_id} is not in range(0, {num_classes})"

    return pred_sem, target_sem, pred_inst, target_inst


def _panoptic_quality_update(
    pred_sem: Tensor,
    target_sem: Tensor,
    pred_inst: Tensor,
    target_inst: Tensor,
    num_classes: int,
    ignore_sem_id: int | None = None,
    min_points: int = 50,
    minkowski: bool = True,
    batch_idx: Tensor | None = None,
    threshold: float = 0.5,
    max_instances: int = 2**20,
    batch_offset: int = 2**10,
):
    pred_sem, target_sem, pred_inst, target_inst = _format_input(
        pred_sem=pred_sem,
        target_sem=target_sem,
        pred_inst=pred_inst,
        target_inst=target_inst,
        num_classes=num_classes,
        ignore_sem_id=ignore_sem_id,
        minkowski=minkowski,
        batch_idx=batch_idx,
        batch_offset=batch_offset,
    )

    device = pred_sem.device

    sem_classes = list(range(0, num_classes))
    if ignore_sem_id is not None:
        # removing the points corresponding to ignore_id from further calculations
        target_sem_not_ignore_mask = target_sem != ignore_sem_id
        pred_sem = pred_sem[target_sem_not_ignore_mask]
        target_sem = target_sem[target_sem_not_ignore_mask]
        pred_inst = pred_inst[target_sem_not_ignore_mask]
        target_inst = target_inst[target_sem_not_ignore_mask]
        sem_classes.remove(ignore_sem_id)

    tp = torch.zeros(num_classes).to(device)
    fp = torch.zeros(num_classes).to(device)
    fn = torch.zeros(num_classes).to(device)
    iou = torch.zeros(num_classes).to(device)
    # process it per class now
    for cl in sem_classes:
        pred_class_mask = pred_sem == cl
        target_class_mask = target_sem == cl

        # get instance points in class (makes outside stuff 0)
        pred_inst_in_class = (
            pred_inst * pred_class_mask.long()
        )  # original api uses np.int64. can switch this to 32 bit
        target_inst_in_class = target_inst * target_class_mask.long()

        unique_pred_iids, counts_pred_iids = torch.unique(
            pred_inst_in_class[pred_inst_in_class > 0], return_counts=True
        )
        id2idx_pred = {id.item(): idx for idx, id in enumerate(unique_pred_iids)}
        matched_pred = torch.tensor([False] * unique_pred_iids.shape[0]).to(device)

        unique_target_iids, counts_target_iids = torch.unique(
            target_inst_in_class[target_inst_in_class > 0], return_counts=True
        )
        id2idx_target = {id.item(): idx for idx, id in enumerate(unique_target_iids)}
        matched_target = torch.tensor([False] * unique_target_iids.shape[0]).to(device)

        # get a boolean mask to get all possible pred id to target id matches while
        # respecting point order
        valid_combos = torch.logical_and(
            pred_inst_in_class > 0, target_inst_in_class > 0
        )
        # get an array of values which can be used to retrieve the corresponding pairs
        # by dividing with max_instances. this will have repetitions for each id pair
        offset_combo = (
            pred_inst_in_class[valid_combos]
            + max_instances * target_inst_in_class[valid_combos]
        )
        # get the unique id pairs, and get the support for those pairs as well
        unique_combo, counts_combo = torch.unique(offset_combo, return_counts=True)

        # retrieve target ids and pred ids in same order as unique_combo
        pot_target_ids = unique_combo // max_instances  # potential target ids
        pot_pred_ids = unique_combo % max_instances  # potential pred ids
        # retrieve the support of corresponding target ids
        pot_target_support = torch.tensor(
            [counts_target_iids[id2idx_target[id.item()]] for id in pot_target_ids]
        ).to(device)
        # same for pred ids
        pot_pred_support = torch.tensor(
            [counts_pred_iids[id2idx_pred[id.item()]] for id in pot_pred_ids]
        ).to(device)
        intersections = counts_combo
        unions = pot_target_support + pot_pred_support - intersections
        # ious for all potential target id pred id matches
        ious = intersections.double() / unions.double()

        tp_indexes = ious > threshold
        tp[cl] += tp_indexes.sum()
        iou[cl] += ious[tp_indexes].sum()

        matched_target[
            [id2idx_target[id.item()] for id in pot_target_ids[tp_indexes]]
        ] = True
        matched_pred[[id2idx_pred[id.item()] for id in pot_pred_ids[tp_indexes]]] = True

        # count the FN
        fn[cl] += torch.logical_and(
            counts_target_iids >= min_points, matched_target == False
        ).sum()

        # count the FP
        fp[cl] += (
            torch.logical_and(counts_pred_iids >= min_points, matched_pred == False)
        ).sum()
    return tp, fp, fn, iou


def _safe_divide(num: Tensor, denom: Tensor, eps: float = 1e-16) -> Tensor:
    """
    Safe division, by preventing division by zero.
    modified torchmetrics version, sets denom to eps instead of 1 when its low.
    Additionally casts to float if input is not already to secure backwards compatibility.
    """
    denom[denom < eps] = eps
    num = num if num.is_floating_point() else num.float()
    denom = denom if denom.is_floating_point() else denom.float()
    return num / denom


# can reuse all the stuff to do sq and rq seperately as well
def _panoptic_quality_compute(
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
    iou: Tensor,
    average: str | None,
    num_classes: int,
    ignore_sem_id: int | None,
):
    # first calculate for all classes
    sq_all = _safe_divide(iou.double(), tp.double())
    rq_all = _safe_divide(
        tp.double(), tp.double() + 0.5 * fp.double() + 0.5 * fn.double()
    )
    pq_all = sq_all * rq_all

    if average == "none" or average is None:
        # we still return a pq for the ignore id so num_classes is maintained
        return pq_all
    elif average == "mean":
        idx = list(range(0, num_classes))
        # if there's an ignore id we dont want that to affect the mean
        if ignore_sem_id is not None:
            idx.remove(ignore_sem_id)
        pq_mean = pq_all[idx].mean()
        return pq_mean
    else:
        raise NotImplementedError


def panoptic_quality(
    pred_sem: Tensor,
    target_sem: Tensor,
    pred_inst: Tensor,
    target_inst: Tensor,
    num_classes: int,
    average: str | None = "mean",
    ignore_id: int | None = None,
    min_points: int = 50,
    minkowski: bool = True,
    batch_idx: Tensor | None = None,
    threshold: float = 0.5,
    max_instances: int = 2**20,
    batch_offset: int = 2**10,
):
    """
    if not minkowski, extra dimensions get flattened into batch dimension.
    see the metric version for more documentation.

    :param average: can be mean or none
    """
    if average is not None:
        assert average.lower() in [
            "none",
            "mean",
        ], f"please pass a proper average and not {average}"
    tp, fp, fn, iou = _panoptic_quality_update(
        pred_sem=pred_sem,
        target_sem=target_sem,
        pred_inst=pred_inst,
        target_inst=target_inst,
        num_classes=num_classes,
        ignore_sem_id=ignore_id,
        min_points=min_points,
        minkowski=minkowski,
        batch_idx=batch_idx,
        threshold=threshold,
        max_instances=max_instances,
        batch_offset=batch_offset,
    )
    return _panoptic_quality_compute(
        tp=tp,
        fp=fp,
        fn=fn,
        iou=iou,
        average=average.lower() if average else None,
        num_classes=num_classes,
        ignore_sem_id=ignore_id,
    )


# stuff classes should not have any instance predictions. set to 0. doesnt matter if
# targets are 0, which they anyways should be
#
# min points
# clusters with min points are still included in tps if they match. this can affect SQ.
# but they are ignored for FN (targets checked) and FP (predictions checked).
# but this is how semantic kitti api already does it and we stick to it
# set min points to be reasonably small for the problem to not have an influence one way
# or the other
#
# SQ and IOU for stuff classes will be the exact same. after setting inst id to 0 for
# stuff, it is bumped by 1 in pq calculation. then the gt instance is 1 for all sem label
# points, and pred instance is 1 for all pred label points. these are matched. and then
# instance intersection is just semantic intersection. union similarly. and instance iou
# is hence semantic iou. and SQ is sum instance iou / num tp. and there is just 1 tp.
