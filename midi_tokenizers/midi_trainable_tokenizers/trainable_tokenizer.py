import os
from typing import Iterable
from abc import abstractmethod

from datasets import Dataset
from tokenizers import Tokenizer, trainers

from midi_tokenizers.base_tokenizers.midi_tokenizer import MidiTokenizer, TokenizerLexicon


class MidiTrainableTokenizer(MidiTokenizer):
    """Base class for trainable MIDI tokenizers with HuggingFace integration."""

    def __init__(
        self,
        lexicon: TokenizerLexicon,
        tokenizer_config: dict,
        base_tokenizer: MidiTokenizer = None,
        text_tokenizer: Tokenizer = None,
        trainer: trainers.Trainer = None,
    ):
        super().__init__(lexicon=lexicon, tokenizer_config=tokenizer_config)
        self.base_tokenizer = base_tokenizer
        self.text_tokenizer = text_tokenizer
        self.trainer = trainer

    @property
    def parameters(self):
        return {
            "lexicon": self.lexicon,
            "tokenizer_config": self.tokenizer_config,
            "base_tokenizer": self.base_tokenizer,
            "text_tokenizer": self.text_tokenizer,
            "trainer": self.trainer,
        }

    @abstractmethod
    def prepare_data_for_training(self, file_name: str, train_dataset: Dataset):
        """Prepare dataset for training text tokenizer.

        Args:
            file_name: Output file path
            train_dataset: Source dataset
        """
        pass

    def train(self, train_dataset: Dataset):
        """Train tokenizer on dataset."""
        file = "tmp_dump.txt"
        open(file, "w").close()
        try:
            self.prepare_data_for_training(file, train_dataset)
            self.text_tokenizer.train([file], trainer=self.trainer)
            self._update_vocab()
        finally:
            os.unlink(file)

    def train_from_text_dataset(self, dataset: Iterable):
        """Train directly from text dataset."""
        self.text_tokenizer.train_from_iterator(dataset, trainer=self.trainer)
        self._update_vocab()

    def _update_vocab(self):
        """Update internal vocab from trained text tokenizer."""
        self.lexicon.token_to_id = self.text_tokenizer.get_vocab()
        self.lexicon.vocab = [token for token in self.lexicon.token_to_id]

    @abstractmethod
    def save_tokenizer(self, path: str):
        """Save tokenizer state to disk.

        Args:
            path: Save location
        """
        pass
