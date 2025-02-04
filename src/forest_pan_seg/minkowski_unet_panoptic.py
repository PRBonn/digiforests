# MIT License
#
# Copyright (c) 2025 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import cuml.cluster
from torch import Tensor
import MinkowskiEngine as ME
import lightning.pytorch as pl

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex


from digiforests_dataloader.utils.logging import logger

from .mlp import MLP
from .metrics import ClasswiseWrapper
from .metrics import PanopticQuality
from .minkunet import CustomMinkUNet14
from .offset_loss import OffsetLoss


class MinkUNetPanoptic(pl.LightningModule):
    """Panoptic Segmentation model using MinkowskiEngine's UNet architecture.

    This model performs both semantic and instance segmentation on 3D point cloud data.
    """

    def __init__(
        self,
        vds: float = 0.1,
        in_channels=1,
        embedding_size=32,
        num_classes=5,
        coord_dimension=3,
        lr: float = 0.001,
        batch_size: int = 1,
        clear_torch_cache=False,
        pq_metrics_every: int = 1,
        dbscan_min_samples: int = 30,  # needs to be the same across runs
    ):
        """Initialize the MinkUNetPanoptic model.

        Args:
            vds (float): Voxel size. Default: 0.1
            in_channels (int): Number of input features per point. Default: 1
            embedding_size (int): Size of embedding vectors. Default: 32
            num_classes (int): Number of semantic classes. Default: 5
            coord_dimension (int): Dimensionality of input coordinates. Default: 3
            lr (float): Initial learning rate. Default: 0.001
            batch_size (int): Batch size for training. Default: 1
            clear_torch_cache (bool): Whether to clear CUDA cache after each step. Default: False
            pq_metrics_every (int): Compute PQ metrics every N epochs. Default: 1
            dbscan_min_samples (int): Minimum samples for DBSCAN clustering. Default: 30

        Note:
            - The `dbscan_min_samples` parameter should remain consistent across runs for reproducibility.
        """
        super().__init__()
        self.save_hyperparameters()
        self.setup_metrics()
        # model setup
        self.mink_unet = CustomMinkUNet14(
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.embedding_size,
            D=self.hparams.coord_dimension,
        )
        self.segmentation_head = MLP(
            self.hparams.embedding_size,
            self.hparams.num_classes,
            batch_norm=True,
            num_layers=2,
        )
        self.instance_head = MLP(
            # TODO: needs to be checked if catting position is useful. and at which resolution
            self.hparams.embedding_size + 3,
            3,  # coord_dimension
            batch_norm=True,
            num_layers=4,
        )
        self.segmentation_loss = torch.nn.CrossEntropyLoss(
            reduction="mean", ignore_index=0
        )  # consider smoothing=True
        self.instance_loss = OffsetLoss(ignore_target_instance_id=0)
        self.last_pq_log = {"train": -1, "val": -1}

    def setup_metrics(self):
        """Initialize and configure metrics for segmentation and panoptic quality evaluation.

        This method sets up MetricCollections for both segmentation and panoptic quality metrics,
        creating separate instances for training, validation, and testing phases.

        Segmentation metrics include:
        - Mean IOU (Intersection over Union)
        - Class-wise IOU for specific labels

        Panoptic quality metrics include:
        - Class-wise Panoptic Quality (PQ) with focus on Tree class

        Note:
        - Segmentation metrics use a ignore index of 0.
        - Panoptic Quality metric ignores class 0 and has a minimum point threshold of 10.
        - Metrics are cloned with specific prefixes for each evaluation phase (train/val/test).
        """
        seg_metrics = MetricCollection(
            {
                "Mean_IOU": MulticlassJaccardIndex(
                    num_classes=self.hparams.num_classes,
                    # TODO: hardcoded ignore index, same in loss
                    ignore_index=0,
                    average="macro",
                ),
                "IOU": ClasswiseWrapper(
                    MulticlassJaccardIndex(
                        num_classes=self.hparams.num_classes,
                        ignore_index=0,
                        average="none",
                    ),
                    labels=["Ignore", "Ground", "Shrub", "Stem", "Canopy"],
                ),
            },
            prefix="seg/",
        )
        pan_metrics = MetricCollection(
            {
                # we're really only interested in Tree PQ. rest are repetitions
                "PQ": ClasswiseWrapper(
                    PanopticQuality(
                        num_classes=self.hparams.num_classes - 1,
                        average="none",
                        ignore_id=0,
                        min_points=10,
                        minkowski=True,
                    ),
                    labels=["Ignore", "Ground", "Shrub", "Tree"],
                ),
            },
            prefix="pan/",
        )

        # unfortunately this is not good enough to infer kwargs from a metric collection of seg and pan metrics. so we have to do it the ugly way for now
        self.train_seg_metrics = seg_metrics.clone(prefix="train/")
        self.val_seg_metrics = seg_metrics.clone(prefix="val/")
        self.test_seg_metrics = seg_metrics.clone(prefix="test/")

        self.train_pan_metrics = pan_metrics.clone(prefix="train/")
        self.val_pan_metrics = pan_metrics.clone(prefix="val/")
        self.test_pan_metrics = pan_metrics.clone(prefix="test/")

    def _should_log_pq(self, mode):
        """essentially a hack to speed up training by not running the clustering necessary for instance metrics every step."""
        if (self.current_epoch - self.last_pq_log[mode]) // (
            self.hparams.pq_metrics_every
        ) >= 1:
            return True
        else:
            return False

    def _logged_pq(self, mode="train"):
        self.last_pq_log[mode] = self.current_epoch

    def forward(self, batch) -> tuple[Tensor, Tensor]:
        """Perform forward pass of the MinkUNetPanoptic model.

        Args:
            batch (dict): Input batch containing:
                - pos (Tensor): Nx4 tensor of coordinates, where the first column is batch_idx
                - intensity (Tensor): NxC tensor of input features

        Returns:
            tuple[Tensor, Tensor]:
                - segmentation_logits: Nx(num_classes) tensor of semantic segmentation logits
                - instance_logits: Nx3 tensor of instance segmentation offsets

        Process:
            1. Quantize input coordinates
            2. Create sparse tensor using MinkowskiEngine
            3. Pass through MinkUNet for feature extraction
            4. Upsample features to original resolution
            5. Generate semantic and instance predictions

        Note:
            Returned logits are at the original input resolution.
        """
        coords = batch["pos"]
        feats = batch["intensity"]
        # quantize the coords - dimensions are batch, x, y, z
        quantization = Tensor(
            [1.0, self.hparams.vds, self.hparams.vds, self.hparams.vds]
        ).to(coords.device)
        coords = torch.div(coords, quantization)

        # alternative to below is to use TensorField to handle the coordinate mapping.
        # see benedikt mersch's MapMOS for an example. here the needs are a bit more complex.
        unique_map, inverse_map = ME.utils.sparse_quantize(
            coords, return_index=True, return_inverse=True, return_maps_only=True
        )
        sparse_coords = coords[unique_map].floor().int()
        # the result for feats is slightly different from TensorField and then sparse.
        # likely which feats they return per voxel is different
        sparse_feats = feats[unique_map]
        predicted_sparse_tensor = self.mink_unet(
            ME.SparseTensor(sparse_feats, sparse_coords)
        )
        sparse_predicted_feats = predicted_sparse_tensor.F
        # upsampling the features to original resolution
        mink_features = sparse_predicted_feats[inverse_map]

        segmentation_logits = self.segmentation_head(mink_features)
        instance_features = torch.cat((mink_features, coords[:, 1:]), dim=1)
        instance_logits = self.instance_head(instance_features)
        return segmentation_logits, instance_logits

    def _get_pq_predictions(
        self,
        seg_logits: Tensor,
        inst_logits: Tensor,
        coords: Tensor,
    ):
        """
        return the pred_sem and pred_inst tensors for use with PanopticQuality
        """
        with torch.no_grad():
            pred_sem = seg_logits.max(dim=1)[1]
            # stem and canopy need to be rolled into tree
            pred_sem[pred_sem == 4] = 3
            # only 4 sem classes now

            offset_coords = coords.clone()

            offset_coords[:, 1:] += inst_logits
            # we only want to cluster the tree points
            # doing anything else is pointless and takes too much compute
            tree_mask = (pred_sem == 3).flatten()
            tree_coords = offset_coords[tree_mask]
            if len(tree_coords):
                tree_cluster_inst = self._cluster_offset_coords(tree_coords)
            else:
                tree_cluster_inst = 0
            # noise from clustering will have noise 0
            noise_inst_mask = tree_cluster_inst == 0
            # set the corresponding predicted semantic class to ignore
            sem_tree_idx_mask = torch.nonzero(tree_mask).squeeze()
            sem_noise_idx_mask = sem_tree_idx_mask[noise_inst_mask]
            pred_sem[sem_noise_idx_mask] = 0
            # recover original length and set inst id for pred_sem stuff classes to 0
            pred_inst = torch.zeros_like(pred_sem).int()
            pred_inst[tree_mask] = tree_cluster_inst

        return pred_sem, pred_inst

    def _cluster_offset_coords(self, tree_coords: Tensor):
        """dbscan using cuML"""
        with torch.no_grad():
            dbscan_eps = self.hparams.vds * 7  # 7 is the answer to life, not 42
            if dbscan_eps >= 1.0:
                # because of the batch index, minimum euclidean distance between two
                # points from different batches will be 1.0
                # eps bounds the maximum distance between two samples for them to be
                # considered neighbors
                # by using this, we can avoid needing to cluster per batch
                logger.warning(
                    f"DSCAN eps {dbscan_eps} means that points across different batches can be clustered together. this is likely not intended"
                )
            cuml_dbscan = cuml.cluster.DBSCAN(
                eps=dbscan_eps, min_samples=self.hparams.dbscan_min_samples
            )
            clustered_instance = cuml_dbscan.fit(tree_coords, out_dtype="int64").labels_
            # dbscan returns with -1 for noise points, we want min 0
            clustered_instance += 1
        return torch.tensor(clustered_instance).to(tree_coords.device).int()

    def training_step(self, batch: dict[str, Tensor], batch_idx, dataloader_idx=0):
        target_sem = batch["semantics"]
        target_inst = batch["instance"]
        target_offset = batch["offset"]
        coords = batch["pos"]

        seg_logits, inst_logits = self.forward(batch)
        seg_loss = self.segmentation_loss(seg_logits, target_sem.long())
        self.log(
            "train/loss_seg",
            seg_loss,
            batch_size=self.hparams.batch_size,
            on_step=True,
            on_epoch=False,
        )
        inst_loss = self.instance_loss(inst_logits, target_offset, target_inst)
        if inst_loss:
            self.log(
                "train/loss_inst",
                inst_loss,
                batch_size=self.hparams.batch_size,
                on_step=True,
                on_epoch=False,
            )
            loss = seg_loss + inst_loss
        else:
            loss = seg_loss
        self.log(
            "train/loss",
            loss,
            batch_size=self.hparams.batch_size,
            on_step=True,
            on_epoch=False,
        )
        # these metrics are epoch level. see the epoch end hooks for compute and log
        self.train_seg_metrics.update(
            preds=seg_logits,
            target=target_sem.int(),
        )

        if self._should_log_pq(mode="train"):
            pred_sem, pred_inst = self._get_pq_predictions(
                seg_logits, inst_logits, coords
            )
            # need to adjust for the change in sem classes in pq eval
            pq_target_sem = target_sem.clone()
            pq_target_sem[pq_target_sem == 4] = 3
            self.train_pan_metrics.update(
                pred_sem=pred_sem,
                target_sem=pq_target_sem,
                pred_inst=pred_inst,
                target_inst=target_inst,
                batch_idx=coords[:, 0],  # minkowski batch coords
            )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        target_sem = batch["semantics"]
        target_inst = batch["instance"]
        target_offset = batch["offset"]
        coords = batch["pos"]

        seg_logits, inst_logits = self.forward(batch)
        seg_loss = self.segmentation_loss(seg_logits, target_sem.long())
        self.log(
            "val/loss_seg",
            seg_loss,
            batch_size=self.hparams.batch_size,
            on_step=True,
            on_epoch=False,
        )
        inst_loss = self.instance_loss(inst_logits, target_offset, target_inst)
        if inst_loss:
            self.log(
                "val/loss_inst",
                inst_loss,
                batch_size=self.hparams.batch_size,
                on_step=True,
                on_epoch=False,
            )
            loss = seg_loss + inst_loss
        else:
            loss = seg_loss
        self.log(
            "val/loss",
            loss,
            batch_size=self.hparams.batch_size,
            on_step=True,
            on_epoch=False,
        )
        # see training_step for some comments
        self.val_seg_metrics.update(
            preds=seg_logits,
            target=target_sem.int(),
        )

        if self._should_log_pq(mode="val"):
            pred_sem, pred_inst = self._get_pq_predictions(
                seg_logits, inst_logits, coords
            )
            # need to adjust for the change in sem classes in pq eval
            pq_target_sem = target_sem.clone()
            pq_target_sem[pq_target_sem == 4] = 3
            self.val_pan_metrics.update(
                pred_sem=pred_sem,
                target_sem=pq_target_sem,
                pred_inst=pred_inst,
                target_inst=target_inst,
                batch_idx=coords[:, 0],  # minkowski batch coords
            )
        return None

    def test_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        """
        Perform a single test step.

        This method is identical to the validation step but is only called when the
        `.test()` method is explicitly invoked on the trainer.
        """
        target_sem = batch["semantics"]
        target_inst = batch["instance"]
        coords = batch["pos"]

        seg_logits, inst_logits = self.forward(batch)
        self.test_seg_metrics.update(
            preds=seg_logits,
            target=target_sem.int(),
        )

        pred_sem, pred_inst = self._get_pq_predictions(seg_logits, inst_logits, coords)
        # need to adjust for the change in sem classes in pq eval
        pq_target_sem = target_sem.clone()
        pq_target_sem[pq_target_sem == 4] = 3

        self.test_pan_metrics.update(
            pred_sem=pred_sem,
            target_sem=pq_target_sem,
            pred_inst=pred_inst,
            target_inst=target_inst,
            batch_idx=coords[:, 0],  # minkowski batch coords
        )

        return {"sem": seg_logits.max(dim=1)[1], "inst": pred_inst}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        coords = batch["pos"]
        seg_logits, inst_logits = self.forward(batch)
        pred = {}
        seg_sem_conf, seg_sem = torch.nn.functional.softmax(seg_logits, dim=1).max(
            dim=1
        )
        pred["seg_sem"] = seg_sem
        pred["seg_sem_conf"] = seg_sem_conf
        pred_sem, pred_inst = self._get_pq_predictions(seg_logits, inst_logits, coords)
        pred["pq_sem"] = pred_sem
        pred["pq_inst"] = pred_inst
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            amsgrad=False,  # might be useful for small dataset
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val/Mean_IOU",
                "frequency": 4,
                "strict": True,
            },
        }

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.hparams.clear_torch_cache:
            # this is horrible. but theres no other way to train on my limited memory gpu
            torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        # we need to manually log at the end of epoch level metrics because we use classwisewrapper
        # see https://github.com/Lightning-AI/torchmetrics/issues/2091
        train_metrics = {}
        train_metrics.update(self.train_seg_metrics.compute())
        if self._should_log_pq(mode="train"):
            train_metrics.update(self.train_pan_metrics.compute())
        self.log_dict(
            train_metrics,
            batch_size=self.hparams.batch_size,
        )
        self.train_seg_metrics.reset()
        if self._should_log_pq(mode="train"):
            self._logged_pq(mode="train")
            self.train_pan_metrics.reset()

    def on_validation_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.hparams.clear_torch_cache:
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        val_metrics = {}
        val_metrics.update(self.val_seg_metrics.compute())
        if self._should_log_pq(mode="val"):
            val_metrics.update(self.val_pan_metrics.compute())
        self.log_dict(
            val_metrics,
            batch_size=self.hparams.batch_size,
        )
        self.val_seg_metrics.reset()
        if self._should_log_pq(mode="val"):
            self._logged_pq(mode="val")
            self.val_pan_metrics.reset()

    def on_test_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.hparams.clear_torch_cache:
            torch.cuda.empty_cache()

    def on_test_epoch_end(self) -> None:
        test_metrics = {}
        test_metrics.update(self.test_seg_metrics.compute())
        test_metrics.update(self.test_pan_metrics.compute())
        self.log_dict(
            test_metrics,
            batch_size=self.hparams.batch_size,
        )
        self.test_seg_metrics.reset()
        self.test_pan_metrics.reset()

    def on_pred_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.hparams.clear_torch_cache:
            torch.cuda.empty_cache()
