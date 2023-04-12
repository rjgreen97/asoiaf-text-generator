import pandas as pd

from src.data.asoiaf_dataset import ASOIAFDataset
from src.data.asoiaf_dataset_builder import ASOIAFDatasetBuilder

dataset_builder = ASOIAFDatasetBuilder(
    "tests/fixtures/test_sentences.json", "gpt2", 0.5
)


def test_init():
    assert dataset_builder.data_filepath == "tests/fixtures/test_sentences.json"
    assert dataset_builder.model_name == "gpt2"
    assert dataset_builder.val_percent == 0.5


def test_build():
    train_dataset, val_dataset = dataset_builder.build()
    assert len(train_dataset) == 12
    assert len(val_dataset) == 12


def test_random_split_dataset():
    df = pd.read_json("tests/fixtures/test_sentences.json")
    df = df.rename(columns={0: "sentences"})
    dataset = ASOIAFDataset("gpt2", df)
    train_dataset, val_dataset = dataset_builder.random_split_dataset(dataset)
    assert train_dataset is not None
    assert val_dataset is not None
    assert train_dataset[0] is not None


if __name__ == "__main__":
    test_init()
    test_build()
    test_random_split_dataset()
