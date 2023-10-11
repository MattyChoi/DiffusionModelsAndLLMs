import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import hydra
import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig


class GenTextModule(L.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(hparams.model)
        self.loss = hydra.utils.instantiate(hparams.loss)

    def forward(self, images: torch.Tensor, *args, **kwargs):
        return self.model(images)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get the training image distribution
        images = batch

        # noise the images randomly and predict the noise from the noisy images
        pred_noise, noise = self.forward(images)

        # computer loss and log
        loss = self.loss(pred_noise, noise)
        self.log("train_loss", loss, on_step=True, one_epoch=True)

        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get the training image distribution
        images = batch

        # noise the images randomly and predict the noise from the noisy images
        pred_noise, noise = self.forward(images)

        # computer loss and log
        loss = self.loss(pred_noise, noise)
        self.log("val_loss", loss, on_step=True)

        return loss

    def validation_epoch_end(self, val_step_outs) -> None:
        tboard = self.logger.experiment

        # sample 16 images
        imgs = self.model.sample(batch_size=16)

        tboard.add_images(f"Generated Images_{self.global_step}", img_tensor=imgs[-1], dataformats="NHWC")


    def test_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get the training image distribution
        images = batch

        # noise the images randomly and predict the noise from the noisy images
        pred_noise, noise = self.forward(images)

        # computer loss and log
        loss = self.loss(pred_noise, noise)
        self.log("test_loss", loss, on_step=True)

        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, list(self.model.parameters())
        )
        lr_scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer)
        return [optimizer], [lr_scheduler]
