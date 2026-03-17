import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Callable
import json


class VisionLanguageDataset(Dataset):
    """
    Dataset for vision-language pre-training.
    Expects a JSONL file with: {"image_path": "...", "caption": "..."}
    """

    def __init__(self, data_path: str, tokenizer, image_transform: Callable, max_length: int = 256):
        self.data = []
        with open(data_path) as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image_tensor = self.image_transform(image)

        encoding = self.tokenizer(
            item["caption"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image": image_tensor,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


class PreferenceDataset(Dataset):
    """
    Dataset for DPO fine-tuning.
    Expects JSONL: {"prompt": "...", "chosen": "...", "rejected": "..."}
    """

    def __init__(self, data_path: str):
        self.data = []
        with open(data_path) as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]
