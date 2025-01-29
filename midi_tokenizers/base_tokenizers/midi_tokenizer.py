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
            self.special_tokens = ["<PAD>", "<CLS>"]
        self.pad_token_id = 0

    @abstractmethod
    def tokenize(self, notes_df: pd.DataFrame) -> list[str]:
        """
        Abstract method to tokenize a record.

        Parameters:
            TODO: Please ALWAYS write consistent abstract methods and documentation

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
        notes_df = self.untokenize(tokens)

        return notes_df

    def encode_notes_df(self, notes_df: pd.DataFrame) -> list[int]:
        notes_df = notes_df.copy()
        note_tokens = self.tokenize(notes_df)
        encoded_notes = self.encode_tokens(note_tokens)

        return encoded_notes

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        encoded_tokens = [self.token_to_id[token] for token in tokens]
        return encoded_tokens

    def pad_to_size(self, token_ids: list[int], target_size: int) -> list[int]:
        padding_size = target_size - len(token_ids)
        if padding_size < 0:
            raise ValueError(
                """
            You requested padding to a size _smaller_ then input sequence.
            This is nonsense and you probably have a bug earlier in the process.
            """
            )
        padding = [self.pad_token_id] * padding_size

        padded_token_ids = token_ids + padding

        return padded_token_ids

    def encode(
        self,
        notes_df: pd.DataFrame,
        pad_to_size: int = 0,
        prefix_tokens: list[str] = [],
    ) -> list[int]:
        notes_df = notes_df.copy()
        tokens = self.tokenize(notes_df)
        encoding = [self.token_to_id[token] for token in tokens]

        padding_size = pad_to_size - len(encoding) - len(prefix_tokens)

        prefix_ids = [self.token_to_id[token] for token in prefix_tokens]
        padding = [self.pad_token_id] * padding_size

        return prefix_ids + encoding + padding

    @classmethod
    def from_dict(cls, tokenizer_desc) -> "MidiTokenizer":
        return cls(**tokenizer_desc["parameters"])

    def to_dict(self) -> dict:
        return {"name": self.name, "parameters": self.parameters}
