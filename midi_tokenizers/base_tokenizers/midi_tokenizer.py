from abc import abstractmethod

import pandas as pd


class MidiTokenizer:
    """Base class for MIDI tokenization.

    Handles conversion between MIDI data and token sequences using configurable vocabulary
    and encoding schemes.

    Attributes:
        lexicon (TokenizerLexicon): Contains vocab, ids and value mappings
        tokenizer_config (dict): Configuration parameters
        name (str): Tokenizer identifier
        pad_token_id (int): ID of padding token
    """

    def __init__(
        self,
        vocab: list[str],
        first_placeholder_id: int,
        tokenizer_config: dict,
    ):
        """Initialize tokenizer with vocabulary and configuration.

        Args:
            tokenizer_config (dict[str, Any]): Args defining tokenization behavior
        """
        self.vocab = vocab
        self.token_to_id = {token: it for it, token in enumerate(vocab)}
        self.first_placeholder_id = first_placeholder_id

        self.tokenizer_config = tokenizer_config
        self.name = "MidiTokenizer"

        self.pad_token_id = self.token_to_id["<PAD>"]

    @classmethod
    @abstractmethod
    def build_tokenizer(cls, tokenizer_config: dict) -> "MidiTokenizer":
        pass

    @classmethod
    @abstractmethod
    def _build_lexicon(cls, tokenizer_config: dict) -> dict:
        pass

    def add_special_tokens(self, special_tokens: list[str]):
        """Add custom tokens by replacing placeholders.

        Args:
            special_tokens (list[str]): New tokens to add
        """
        for special_token in special_tokens:
            # Remove placeholder definition
            placeholder_token = self.vocab[self.first_placeholder_id]
            self.token_to_id.pop(placeholder_token)

            # Switch the placeholder token for a special token
            self.vocab[self.first_placeholder_id] = special_token
            self.token_to_id[special_token] = self.first_placeholder_id
            self.first_placeholder_id += 1

    @abstractmethod
    def tokenize(self, notes_df: pd.DataFrame) -> list[str]:
        """
        Abstract method to tokenize a record.

        Args:
            notes_df: DataFrame with columns [start, end, pitch, velocity]

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

        Args:
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
        return len(self.lexicon.vocab)

    def decode(self, token_ids: list[int]) -> pd.DataFrame:
        """
        Decodes a list of token IDs into a DataFrame.

        Args:
            token_ids (list[int]): List of token IDs to decode.

        Returns:
            pd.DataFrame: DataFrame representation of the decoded tokens.
        """
        tokens = [self.lexicon.vocab[token_id] for token_id in token_ids]
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
