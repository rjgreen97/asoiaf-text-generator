import pandas as pd
from torch.utils.data import random_split

from src.data.asoiaf_dataset import ASOIAFDataset


class ASOIAFDatasetBuilder:
    def __init__(self, data_filepath, model_name, val_percent=0.2):
        self.data_filepath = data_filepath
        self.model_name = model_name
        self.val_percent = val_percent

    def build(self):
        print(f"Reading data from specified filepath: {self.data_filepath}")
        df = pd.read_json(self.data_filepath)
        df = df.rename(columns={0: "sentences"})
        dataset = ASOIAFDataset(self.model_name, df)
        train_dataset, val_dataset = self.random_split_dataset(dataset)
        return train_dataset, val_dataset

    def random_split_dataset(self, dataset):
        val_size = int(len(dataset) * self.val_percent)
        train_size = len(dataset) - val_size
        return random_split(dataset, [train_size, val_size])
