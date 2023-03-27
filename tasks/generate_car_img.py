import hydra
import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import DictConfig


class DiffusionModule(pl.LightningModule):
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

        return self.loss(pred_noise, noise)

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get the training image distribution
        images = batch

        # noise the images randomly and predict the noise from the noisy images
        pred_noise, noise = self.forward(images)

        return self.loss(pred_noise, noise)

    def test_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get the training image distribution
        images = batch

        # noise the images randomly and predict the noise from the noisy images
        pred_noise, noise = self.forward(images)

        return self.loss(pred_noise, noise)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, list(self.model.parameters())
        )
        lr_scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer)
        return [optimizer], [lr_scheduler]
