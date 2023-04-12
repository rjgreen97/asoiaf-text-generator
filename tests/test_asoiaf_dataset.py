import pandas as pd

from src.data.asoiaf_dataset import ASOIAFDataset

test_df = pd.read_json("tests/fixtures/test_sentences.json")
df = test_df.rename(columns={0: "sentences"})
dataset = ASOIAFDataset("gpt2", df)


def test_init():
    assert dataset.model_name == "gpt2"
    assert dataset.df.equals(df)
    assert dataset.max_length == 256
    assert dataset.bos_token == "<|beginning_of_sentence|>"
    assert dataset.eos_token == "<|end_of_sentence|>"
    assert dataset.pad_token == "<|pad|>"
    assert dataset.tokenizer is not None


def test_len():
    assert len(dataset) == len(df)


def test_getitem():
    assert dataset[2]["sentence"] == "It was here the ravens came, after long flight."
    assert len(dataset[2]["input_ids"]) == 256
    assert len(dataset[2]["attention_mask"]) == 256


def test_tokens():
    assert (
        dataset.tokenizer.decode(dataset[2]["input_ids"][:14])
        == "<|beginning_of_sentence|>It was here the ravens came, after long flight.<|end_of_sentence|>"
    )


def test_encode_text():
    dataset = ASOIAFDataset("gpt2", df, max_lenth=24)
    encodings_dict = dataset.encode_text("I am the big man.")
    assert len(encodings_dict["input_ids"]) == 24
    assert len(encodings_dict["attention_mask"]) == 24


if __name__ == "__main__":
    test_init()
    test_len()
    test_getitem()
    test_tokens()
    test_encode_text()
