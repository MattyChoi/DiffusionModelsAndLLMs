import os
import sys
from typing import Iterator, List, Any
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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


# Since the dataset is too large, we make it into an iterable dataset
class OSCARCorpus(IterableDataset):
    """
    Dataset of texts in english from oscar-corpus/OSCAR-2301 huggingface
    """
    def __init__(self, data_dir="", transform=None):
        super(OSCARCorpus, self).__init__()
        
        self.dataset = load_dataset(
            "oscar-corpus/OSCAR-2301",
            use_auth_token="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", # required
            language="en", 
            streaming=True, # optional
            split="train", # optional, but the dataset only has a train split
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-cased",
            use_fast=True,
            use_auth_token=False,
        )
        
    def __iter__(self) -> Iterator:
        item = iter(self.dataset)

        text = self.tokenizer(item["text"], padding="max_length", truncation=True,)# max_length=self.max_seq_length)
        id = item["id"]
        return {
            text: text,
            id: id,
        }
    
    
# class OSCARCorpus(Dataset):
#     """
#     Dataset of car images
#     """
#     def __init__(self, data_dir="", transform=None):
#         super(OSCARCorpus, self).__init__()
        
#         self.dataset = load_dataset(
#             "oscar-corpus/OSCAR-2301",
#             use_auth_token=True, # required
#             language="en", 
#             split="train",
#         ) # optional, but the dataset only has a train split
        
#         self.vectorizer = AutoTokenizer.from_pretrained()
        
#     def __getitem__(self, index):
#         return self.dataset

#     def __len__(self):
#         return len(self.dataset)

#     def collate_fn(self, batch):
#         return torch.stack(batch, dim=0)
    

if __name__ == "__main__":
    print('hi')
    ds = OSCARCorpus()