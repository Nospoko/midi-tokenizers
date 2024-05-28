import os
from typing import Iterable
from abc import abstractmethod

from datasets import Dataset
from tokenizers import Tokenizer, trainers

from midi_tokenizers.midi_tokenizer import MidiTokenizer


class MidiTrainableTokenizer(MidiTokenizer):
    """
    Base class for trainable MIDI tokenizers.

    Inherits from MidiTokenizer and adds functionality for training the tokenizer.

    Attributes:
        base_tokenizer (MidiTokenizer): The base tokenizer used for generating initial tokens.
        text_tokenizer (Tokenizer): The text tokenizer used for training.
        trainer (trainers.Trainer): The trainer used for training the text tokenizer.
    """

    def __init__(self, special_tokens: list[str] = None):
        """
        Initialize the MidiTrainableTokenizer with optional special tokens.

        Parameters:
            special_tokens (list[str], optional): List of special tokens. Defaults to None.
        """
        super().__init__(special_tokens=special_tokens)
        self.base_tokenizer: MidiTokenizer = None
        self.text_tokenizer: Tokenizer = None
        self.trainer: trainers.Trainer = None

    @abstractmethod
    def prepare_data_for_training(file_name: str, train_dataset: Dataset):
        """
        Abstract method to prepare data for training.

        This method should be implemented by subclasses to specify how the training data
        should be prepared and written to a file.

        Parameters:
            file_name (str): The name of the file to write the prepared data.
            train_dataset (Dataset): The dataset to prepare for training.
        """
        pass

    def train(self, train_dataset: Dataset):
        """
        Train the tokenizer using the provided training dataset.

        Parameters:
            train_dataset (Dataset): The dataset to use for training.
        """
        file = "tmp_dump.txt"
        # create an empy file
        open(file, "w").close()
        try:
            self.prepare_data_for_training(file_name=file, train_dataset=train_dataset)

            self.text_tokenizer.train([file], trainer=self.trainer)
            self.vocab = self.text_tokenizer.get_vocab()
        finally:
            # make sure to always clean up
            os.unlink(file)

    def train_from_text_dataset(self, dataset: Iterable):
        """
        Train the tokenizer directly from a text dataset.

        Parameters:
            dataset (Iterable): An iterable of text data to use for training.
        """
        self.text_tokenizer.train_from_iterator(dataset, trainer=self.trainer)

        self.vocab = self.text_tokenizer.get_vocab()

    @abstractmethod
    def save_tokenizer(self, path: str):
        """
        Abstract method to save the tokenizer to a specified path.

        This method should be implemented by subclasses to specify how the tokenizer
        should be saved to disk.

        Parameters:
            path (str): The path to save the tokenizer.
        """
        pass
