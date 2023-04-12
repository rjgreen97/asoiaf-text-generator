import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.data.asoiaf_dataset_builder import ASOIAFDatasetBuilder
from src.training.training_session_arg_parser import TrainingSessionArgParser


class TrainingSession:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            bos_token="<|beginning_of_sentence|>",
            eos_token="<|end_of_sentence|>",
            pad_token="<|pad|>",
        )

    def run(self):
        self.create_datasets()
        self.create_dataloaders()
        self.create_model()
        self.create_trainer()
        self.trainer.train()

    def create_datasets(self):
        dataset_builder = ASOIAFDatasetBuilder(
            self.args.data_filepath, self.args.model_name, self.args.val_percent
        )
        self.train_dataset, self.val_dataset = dataset_builder.build()

    def create_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size
        )

    def create_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def create_trainer(self):
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        training_args = TrainingArguments(
            output_dir=f"./models/{self.args.model_name}/{time}",
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.batch_size,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=lambda data: {
                "input_ids": torch.stack([f["input_ids"] for f in data]),
                "attention_mask": torch.stack([f["attention_mask"] for f in data]),
                "labels": torch.stack([f["input_ids"] for f in data]),
            },
        )


if __name__ == "__main__":
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(args)
    session.run()
