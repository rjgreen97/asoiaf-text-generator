from argparse import ArgumentParser


class TrainingSessionArgParser:
    def __init__(self):
        self.parser = ArgumentParser(description='Training session arguments')
        self.add_args()

    def add_args(self):
        pass

    def parse_args(self):
        return self.parser.parse_args()
