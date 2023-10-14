import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import hydra
import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig


class TextGenerationModule(L.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(hparams.model)
        # self.loss = hydra.utils.instantiate(hparams.loss)

    def training_step(
        self, batch, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get all the data
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["input_ids"]

        # run it through the model to get the logits and loss
        logits, loss = self.model(input_ids, attn_mask, labels)

        # log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get all the data
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["input_ids"]

        # run it through the model to get the logits and loss
        logits, loss = self.model(input_ids, attn_mask, labels)

        # log the loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    # def on_validation_epoch_end(self) -> None:
    #     tboard = self.logger.experiment

    #     # sample 16 images
    #     imgs = self.model.sample(batch_size=16)

    #     tboard.add_images(f"Generated Images_{self.global_step}", img_tensor=imgs[-1], dataformats="NHWC")


    def test_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get all the data
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["input_ids"]

        # run it through the model to get the logits and loss
        logits, loss = self.model(input_ids, attn_mask, labels)

        # log the loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, list(self.model.parameters())
        )
        lr_scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer)
        return [optimizer], [lr_scheduler]
