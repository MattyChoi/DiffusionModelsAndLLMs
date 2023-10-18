import os
import sys
from functools import partial
from typing import List, Any
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


def tokenize_fn(
    examples: List[dict],
    max_length: int,
    tokenizer: Any,
    text_attr: str = "text",
    padding: str = "max_length",
):
    return tokenizer(
        examples[text_attr],
        add_special_tokens=False,
        padding=padding,
        max_length=max_length,
        truncation=True,
    )

# load the dataset from huggingface, we use the e2e_nlg dataset from
# huggingface here https://huggingface.co/datasets/e2e_nlg
class NLGDataset(Dataset):
    """
    Dataset of texts in english from oscar-corpus/OSCAR-2301 huggingface
    """
    def __init__(self, split="train", tokenizer="", max_length=1024, transform=None):
        super(NLGDataset, self).__init__()
        
        dataset = load_dataset(
            "e2e_nlg",
            split=split,
        )
        
        # create a tokenizer here, we use the bert-large-cased tokenizer from huggingface
        # but we can also pretrain one
        self.tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            use_fast=True,
            use_auth_token=False,
            pad_token="<|endoftext|>",
            padding_side="left"
        ) if not tokenizer else tokenizer

        # tokenize the dataset
        tokenized_dataset = dataset.map(
            partial(
                tokenize_fn,
                max_length=max_length,
                tokenizer=self.tokenizer,
                text_attr="human_reference",
            ),
            batched=True,
            num_proc=1,
        ).rename_column("human_reference", "text")

        # only use the input_id columns
        tokenized_dataset.set_format("pt", columns=["input_ids", "attention_mask"])

        self.dataset = tokenized_dataset

        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt",
        )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):
        return self.collator(batch)
    

# if __name__ == "__main__":
#     ds = NLGDataset()

#     from torch.utils.data import DataLoader

#     dataloader = DataLoader(
#         ds,
#         batch_size=4,
#         drop_last=True,
#         collate_fn=ds.collate_fn,
#         num_workers=1,
#         pin_memory=True,
#     )

#     print('start')
#     print(next(iter(dataloader)))