import json

import tokenizers
import pandas as pd
from datasets import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from midi_tokenizers.base_tokenizers.midi_tokenizer import MidiTokenizer
from midi_tokenizers.midi_trainable_tokenizers.trainable_tokenizer import MidiTrainableTokenizer
import midi_tokenizers.midi_tokenizers_generation.base_tokenizer_generator as base_tokenizer_generator


class AwesomeMidiTokenizer(MidiTrainableTokenizer):
    """
    A MIDI tokenizer that uses BPE and encodes base tokenizer token IDs as characters.

    Inherits from MidiTrainableTokenizer and uses a byte-pair encoding (BPE) tokenizer for MIDI token sequences.

    Attributes:
        base_tokenizer (MidiTokenizer): The base tokenizer used for generating initial tokens.
        text_tokenizer (Tokenizer): The BPE tokenizer used for training and tokenization.
        name (str): Name of the tokenizer.
        max_vocab_size (int): Maximum size of the vocabulary.
        max_token_length (int): Maximum length of tokens.
        special_tokens (list[str]): List of special tokens.
        encryption_offset (int): Offset used when converting token IDs to characters.
    """

    def __init__(
        self,
        base_tokenizer: MidiTokenizer,
        bpe_tokenizer: Tokenizer = None,
        max_vocab_size: int = 30000,
        max_token_length: int = 128,
        special_tokens: list[str] = None,
    ):
        """
        Initialize the AwesomeMidiTokenizer with a base tokenizer and optional BPE tokenizer and vocabulary size.

        Parameters:
            base_tokenizer (MidiTokenizer): The base tokenizer used for generating initial tokens.
            bpe_tokenizer (Tokenizer, optional): The BPE tokenizer. If None, a new BPE tokenizer is created.
            Defaults to None.
            max_vocab_size (int, optional): Maximum size of the vocabulary. Defaults to 30000.
            max_token_length (int, optional): Maximum length of tokens. Defaults to 128.
            special_tokens (list[str], optional): List of special tokens. Defaults to None.
        """
        # This is a tricky tokenizer: it uses base_tokenizer token IDs as characters.
        # encryption_offset is used when converting token IDs to characters:
        # we do not want to use a NULL, '\n', nor " " character (because it splits the words)
        self.encryption_offset = 100

        super().__init__(special_tokens=special_tokens)
        self.base_tokenizer = base_tokenizer
        self.text_tokenizer = bpe_tokenizer
        self.name = "AwesomeMidiTokenizer"
        self.max_vocab_size = max_vocab_size
        self.max_token_length = max_token_length

        if self.text_tokenizer is None:
            # Initialize empty tokenizer and a trainer
            self.text_tokenizer = Tokenizer(model=models.BPE())
            self.text_tokenizer.pre_tokenizer = self.prepare_text_pre_tokenizer()
            self.trainer = trainers.BpeTrainer(
                vocab_size=self.max_vocab_size,
                max_token_length=self.max_token_length,
                special_tokens=self.special_tokens,
            )

        # Initialize token-to-ID mapping - huggingface vocab is like our token_to_id
        self.token_to_id = self.text_tokenizer.get_vocab()
        self.vocab = [token for token, it in self.token_to_id.items()]

    def awesome_tokens_to_base_ids(self, awesome_tokens: list[str]) -> list[int]:
        """
        Convert awesome tokens to base token IDs.

        Parameters:
            awesome_tokens (list[str]): List of awesome tokens to convert.

        Returns:
            list[int]: List of base token IDs.
        """
        base_token_ids = []
        for awesome_token in awesome_tokens:
            for character in awesome_token:
                base_token_id = ord(character) - self.encryption_offset
                base_token_ids.append(base_token_id)
        return base_token_ids

    def base_ids_to_awesome_tokens(self, base_token_ids: list[int]) -> list[str]:
        """
        Convert base token IDs to awesome tokens.

        Parameters:
            base_token_ids (list[int]): List of base token IDs to convert.

        Returns:
            list[str]: List of awesome tokens.
        """
        awesome_tokens = []
        for token_id in base_token_ids:
            char = chr(token_id + self.encryption_offset)
            awesome_tokens.append(char)
        return awesome_tokens

    def prepare_data_for_training(self, file_name: str, train_dataset: Dataset):
        """
        Prepare data for training by converting MIDI notes to tokens and writing them to a file.

        Parameters:
            file_name (str): The name of the file to write the prepared data.
            train_dataset (Dataset): The dataset to prepare for training.
        """

        def process_record(record):
            notes = pd.DataFrame(record["notes"])
            tokens = self.base_tokenizer.encode(notes=notes)
            awesome_tokens = self.base_ids_to_awesome_tokens(tokens)

            # Split tokens into chunks of less than max_token_length characters
            chunked_tokens = []
            for i in range(0, len(tokens), self.max_token_length):
                chunk = "".join(str(token) for token in awesome_tokens[i : i + self.max_token_length])
                chunked_tokens.append(chunk)

            # Join chunks with whitespace
            return " ".join(chunked_tokens) + "\n"

        with open(file=file_name, mode="w+", encoding="utf-8") as file:
            for record in train_dataset:
                result = process_record(record=record)
                file.write(result)

    def prepare_text_pre_tokenizer(self) -> pre_tokenizers.PreTokenizer:
        """
        Prepare the pre-tokenizer for text tokenization.

        Returns:
            pre_tokenizers.PreTokenizer: The prepared pre-tokenizer.
        """
        whitespace_tokenizer = pre_tokenizers.WhitespaceSplit()

        # In the txt file, new records begin with a newline
        end_line_splitter = pre_tokenizers.Split("\n", behavior="removed")

        text_pre_tokenizers = [
            end_line_splitter,
            whitespace_tokenizer,
        ]
        return pre_tokenizers.Sequence(text_pre_tokenizers)

    def tokenize(self, notes: pd.DataFrame) -> list[str]:
        """
        Tokenize MIDI notes using the base tokenizer and BPE tokenizer.

        Parameters:
            notes (pd.DataFrame): DataFrame of MIDI notes to tokenize.

        Returns:
            list[str]: List of tokens.
        """
        base_token_ids = self.base_tokenizer.encode(notes)
        awesome_tokens = self.base_ids_to_awesome_tokens(base_token_ids)
        concatenated_tokens = "".join(awesome_tokens)

        encoding: tokenizers.Encoding = self.text_tokenizer.encode(concatenated_tokens)
        return encoding.tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        """
        Convert tokens back into MIDI notes.

        Parameters:
            tokens (list[str]): List of tokens to untokenize.

        Returns:
            pd.DataFrame: DataFrame of untokenized MIDI notes.
        """
        base_token_ids = self.awesome_tokens_to_base_ids(tokens)
        base_tokens = [self.base_tokenizer.lexicon[base_token_id] for base_token_id in base_token_ids]
        return self.base_tokenizer.untokenize(tokens=base_tokens)

    def to_dict(self):
        tokenizer_desc = {
            "base_tokenizer": self.base_tokenizer.name,
            "base_tokenizer_parameters": self.base_tokenizer.parameters,
            "bpe_tokenizer": self.text_tokenizer.to_str(),
        }
        return tokenizer_desc

    def save_tokenizer(self, path: str):
        """
        Save the tokenizer to a specified path.

        Parameters:
            path (str): The path to save the tokenizer.
        """
        tokenizer_desc = self.to_dict()
        with open(path, "w+") as f:
            json.dump(tokenizer_desc, f)

    @classmethod
    def from_dict(cls, tokenizer_desc: str) -> "AwesomeMidiTokenizer":
        base_tokenizer_name = tokenizer_desc["base_tokenizer"]
        parameters = tokenizer_desc["base_tokenizer_parameters"]
        bpe_tokenizer_json = tokenizer_desc["bpe_tokenizer"]

        base_tokenizer = base_tokenizer_generator.generate_tokenizer(
            name=base_tokenizer_name,
            parameters=parameters,
        )
        tokenizer = Tokenizer.from_str(bpe_tokenizer_json)
        return cls(base_tokenizer=base_tokenizer, bpe_tokenizer=tokenizer)

    @classmethod
    def from_file(cls, path: str) -> "AwesomeMidiTokenizer":
        """
        Load an AwesomeMidiTokenizer from a specified file.

        Parameters:
            path (str): The path to load the tokenizer from.

        Returns:
            AwesomeMidiTokenizer: The loaded AwesomeMidiTokenizer.
        """
        with open(path, "r") as f:
            tokenizer_desc = json.load(f)

        return AwesomeMidiTokenizer.from_dict(tokenizer_desc=tokenizer_desc)
