from abc import abstractmethod

import pandas as pd


class MidiTokenizer:
    """
    Base class for MIDI tokenizers.

    Attributes:
        token_to_id (dict): Mapping from tokens to their corresponding IDs.
        vocab (list): List of tokens in the vocabulary.
        name (str): Name of the tokenizer.
        special_tokens (list): List of special tokens.
    """

    def __init__(self, special_tokens: list[str] = None):
        """
        Initializes the MidiTokenizer with optional special tokens.

        Parameters:
            special_tokens (list[str], optional): List of special tokens. Defaults to None.
        """
        self.token_to_id = None
        self.vocab = []
        self.name = "MidiTokenizer"
        self.special_tokens = special_tokens
        if self.special_tokens is None:
            self.special_tokens = ["<CLS>"]

    @abstractmethod
    def tokenize(self, record: dict) -> list[str]:
        """
        Abstract method to tokenize a record.

        Parameters:
            record (dict): The input record to tokenize.

        Returns:
            list[str]: List of tokens.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Your encoder needs *tokenize* implementation")

    @abstractmethod
    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        """
        Abstract method to untokenize a list of tokens.

        Parameters:
            tokens (list[str]): The list of tokens to untokenize.

        Returns:
            pd.DataFrame: DataFrame representation of the untokenized tokens.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Your encoder needs *untokenize* implementation")

    @property
    def parameters(self):
        return {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def decode(self, token_ids: list[int]) -> pd.DataFrame:
        """
        Decodes a list of token IDs into a DataFrame.

        Parameters:
            token_ids (list[int]): List of token IDs to decode.

        Returns:
            pd.DataFrame: DataFrame representation of the decoded tokens.
        """
        tokens = [self.vocab[token_id] for token_id in token_ids]
        df = self.untokenize(tokens)

        return df

    def encode(self, notes: pd.DataFrame) -> list[int]:
        """
        Encodes a DataFrame of notes into a list of token IDs.

        Parameters:
            notes (pd.DataFrame): DataFrame of notes to encode.

        Returns:
            list[int]: List of token IDs.
        """
        notes = notes.copy()
        tokens = self.tokenize(notes)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids
