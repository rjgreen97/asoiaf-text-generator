from argparse import ArgumentParser


class TrainingSessionArgParser:
    def __init__(self):
        self.parser = ArgumentParser(description="Training session arguments")
        self.add_args()

    def add_args(self):
        self.parser.add_argument(
            "--data_filepath",
            type=str,
            default="data/processed/all_books_paragraphs.json",
            help="Path to the json file containing text data",
        )
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="facebook/opt-125m",
            help="HuggingFace CausalLM pre-trained model to be fine tuned",
        )
        self.parser.add_argument(
            "--num_epochs",
            type=int,
            default=64,
            help="Number of epochs to train the model",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Batch size to be used during training",
        )
        self.parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Learning rate to be used during training",
        )
        self.parser.add_argument(
            "--val_percent",
            type=float,
            default=0.2,
            help="Percentage of the dataset to be used for validation",
        )

    def parse_args(self):
        return self.parser.parse_args()
