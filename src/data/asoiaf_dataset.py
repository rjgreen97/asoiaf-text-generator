import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ASOIAFDataset(Dataset):
    def __init__(self, model_name, df, max_lenth=256):
        self.model_name = model_name
        self.df = df
        self.max_length = max_lenth
        self.bos_token = "<|beginning_of_sentence|>"
        self.eos_token = "<|end_of_sentence|>"
        self.pad_token = "<|pad|>"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.df["sentences"][index]
        encodings_dict = self.encode_text(sentence)
        return {
            "sentence": sentence,
            "input_ids": torch.tensor(encodings_dict["input_ids"]),
            "attention_mask": torch.tensor(encodings_dict["attention_mask"]),
        }

    def encode_text(self, text):
        return self.tokenizer(
            self.bos_token + text + self.eos_token,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
