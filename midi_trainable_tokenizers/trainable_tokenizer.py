import tempfile
from typing import Iterable
from abc import abstractmethod

from datasets import Dataset
from tokenizers import Tokenizer, trainers

from midi_tokenizers.midi_tokenizer import MidiTokenizer


class MidiTrainableTokenizer(MidiTokenizer):
    def __init__(self):
        self.base_tokenizer: MidiTokenizer = None
        self.tokenizer: Tokenizer = None
        self.trainer: trainers.Trainer = None

    @abstractmethod
    def prepare_data_for_training(file_name: str, train_dataset: Dataset):
        pass

    def train(self, train_dataset: Dataset):
        file = tempfile.NamedTemporaryFile()
        self.prepare_data_for_training(file_name=file.name, train_dataset=train_dataset)

        self.tokenizer.train([file.name], trainer=self.trainer)
        self.vocab = self.tokenizer.get_vocab()

    def train_from_text_dataset(self, dataset: Iterable):
        self.tokenizer.train_from_iterator(dataset, trainer=self.trainer)

        self.vocab = self.tokenizer.get_vocab()

    @abstractmethod
    def save_tokenizer(self, path: str):
        pass
