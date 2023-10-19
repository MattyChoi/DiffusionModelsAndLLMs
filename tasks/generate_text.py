import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import hydra
from omegaconf import DictConfig
import lightning as L
import torch
import torchmetrics
from transformers import AutoTokenizer


class TextGenerationModule(L.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(hparams.model)
        # self.loss = hydra.utils.instantiate(hparams.loss)


    def generate(self, prompt):
        # tboard = self.logger.experiment
        max_new_tokens = 500

        input = self.model.tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=self.model.max_length,
            truncation=True,
        )
        start_ids = input["input_ids"]
        attn_mask = input["attention_mask"]

        start_ids = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        attn_mask = torch.tensor(attn_mask, dtype=torch.float, device=self.device)[None, ...]

        answer = self.model.generate(start_ids, max_new_tokens, attn_mask)
        self.print(self.model.tokenizer.decode(answer[0].tolist()))
        self.print('------------------------------------------')


    def training_step(
        self, batch, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get all the data
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["labels"]
        
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
        labels = batch["labels"]

        # run it through the model to get the logits and loss
        logits, loss = self.model(input_ids, attn_mask, labels)

        # log the loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    

    def on_validation_epoch_end(self) -> None:
        prompt = "What is the answer to life, the universe, and everything?"
        self.generate(prompt)


    def test_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get all the data
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["labels"]

        # run it through the model to get the logits and loss
        logits, loss = self.model(input_ids, attn_mask, labels)

        # log the loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def on_test_epoch_end(self) -> None:
        prompt = "What is the answer to life, the universe, and everything?"
        self.generate(prompt)


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, list(self.model.parameters())
        )
        lr_scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer)
        return [optimizer], [lr_scheduler]
