from abc import abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class TokenizerLexicon:
    """Storage for tokenizer's vocabulary and mappings.

    Attributes:
        vocab: All available tokens
        first_placeholder_id: Starting ID for placeholder tokens
        token_to_id: Maps tokens to their numeric IDs
        token_to_value: Maps tokens to their semantic values (pitch/velocity/time steps)
    """

    vocab: list[str]
    first_placeholder_id: int
    token_to_id: dict[str, int]
    token_to_value: int


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

    def __init__(self, lexicon: TokenizerLexicon, tokenizer_config: dict):
        """Initialize tokenizer with vocabulary and configuration.

        Args:
            lexicon (TokenizerLexicon): Contains vocabulary, token/ID mappings and placeholder info
            tokenizer_config (dict[str, Any]): Args defining tokenization behavior
        """
        self.token_to_id = None
        self.lexicon = lexicon
        self.tokenizer_config = tokenizer_config
        self.name = "MidiTokenizer"
        self.pad_token_id = lexicon.vocab.index("<PAD>")

    @classmethod
    def build_tokenizer(cls, tokenizer_config: dict) -> "MidiTokenizer":
        print(cls)
        lexicon = cls._build_lexicon(tokenizer_config=tokenizer_config)

        tokenizer = cls(
            lexicon=lexicon,
            tokenizer_config=tokenizer_config,
        )
        return tokenizer

    # Each tokenizer can have its own way of building vocab
    @classmethod
    @abstractmethod
    def _build_lexicon(cls, tokenizer_config: dict) -> TokenizerLexicon:
        pass

    def add_special_tokens(self, special_tokens: list[str]):
        """Add custom tokens by replacing placeholders.

        Args:
            special_tokens (list[str]): New tokens to add
        """
        for special_token in special_tokens:
            # Remove placeholder definition
            placeholder_token = self.lexicon.vocab[self.lexicon.first_placeholder_id]
            self.lexicon.token_to_id.pop(placeholder_token)

            # Switch the placeholder token for a special token
            self.lexicon.vocab[self.lexicon.first_placeholder_id] = special_token
            self.lexicon.token_to_id[special_token] = self.lexicon.first_placeholder_id
            self.lexicon.first_placeholder_id += 1

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
