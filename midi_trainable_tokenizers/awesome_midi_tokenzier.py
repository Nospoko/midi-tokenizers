import json

import tokenizers
import pandas as pd
from datasets import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from midi_tokenizers.midi_tokenizer import MidiTokenizer
from midi_trainable_tokenizers.trainable_tokenizer import MidiTrainableTokenizer
from midi_tokenizers_generation.base_tokenizer_generator import generate_tokenizer as generate_base_tokenizer


class AwesomeMidiTokenizer(MidiTrainableTokenizer):
    def __init__(
        self,
        base_tokenizer: MidiTokenizer,
        bpe_tokenizer: Tokenizer = None,
        max_vocab_size: int = None,
        max_token_length: int = 128,
    ):
        # this is a tricky tokenizer : it uses base_tokenizer token ids as characters.
        # encryption_offset is used when converting token ids to characters:
        # we do not want to use a NULL character nor "\n" (because it splits the words - i might change it later)
        # nor " " - because we need to have WhitespaceSplit pre-tokenizer to serialize
        # huggingface tokenizers correctly (no idea why)
        self.encryption_offset = 100

        super().__init__()
        self.base_tokenizer = base_tokenizer

        self.text_tokenizer = bpe_tokenizer
        self.name = "AwesomeMidiTokenizer"
        self.max_vocab_size = max_vocab_size
        self.max_token_length = max_token_length
        if self.max_vocab_size is None:
            self.max_vocab_size = 30000  # default BpeTrainer vocab_size

        if self.text_tokenizer is None:
            # Initialize empty tokenizer and a trainer
            self.text_tokenizer = Tokenizer(model=models.BPE())
            self.text_tokenizer.pre_tokenizer = self.prepare_text_pre_tokenizer()
            self.trainer = trainers.BpeTrainer(
                vocab_size=self.max_vocab_size,
                max_token_length=self.max_token_length,
                special_tokens=["<CLS>"],
            )

        self.vocab = self.text_tokenizer.get_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def awesome_tokens_to_base_ids(self, awesome_tokens: list[str]) -> list[int]:
        base_token_ids = []
        for awesome_token in awesome_tokens:
            for character in awesome_token:
                base_token_id = ord(character) - self.encryption_offset
                base_token_ids.append(base_token_id)
        return base_token_ids

    def base_ids_to_awesome_tokens(self, base_token_ids: list[int]) -> list[str]:
        awesome_tokens = []

        for token_id in base_token_ids:
            char = chr(token_id + self.encryption_offset)
            awesome_tokens.append(char)

        return awesome_tokens

    def prepare_data_for_training(self, file_name: str, train_dataset: Dataset):
        def process_record(record):
            notes = pd.DataFrame(record["notes"])
            tokens = self.base_tokenizer.encode(notes=notes)
            awesome_tokens = self.base_ids_to_awesome_tokens(tokens)

            # Split tokens into chunks of less than max_token_length characters
            # Create chunks of self.max_token_length characters
            chunked_tokens = []
            chunk = ""
            for i in range(0, len(tokens), self.max_token_length):
                chunk = "".join(str(token) for token in awesome_tokens[i : i + self.max_token_length])
                chunked_tokens.append(chunk)

            # Join chunks with whitespace
            return " ".join(chunked_tokens) + "\n"

        with open(file=file_name, mode="w+", encoding="utf-8") as file:
            for record in train_dataset:
                result = process_record(record=record)
                file.write(result)

    def prepare_text_pre_tokenizer(self):
        # We have to use this - we cannot load saved tokenizer otherwise
        whitespace_tokenzier = pre_tokenizers.WhitespaceSplit()

        # In the txt file, new records begin with a newline
        end_line_splitter = pre_tokenizers.Split("\n", behavior="removed")

        text_pre_tokenizers = [
            end_line_splitter,
            whitespace_tokenzier,
        ]
        return pre_tokenizers.Sequence(text_pre_tokenizers)

    def tokenize(self, notes: pd.DataFrame) -> list[str]:
        base_token_ids = self.base_tokenizer.encode(notes)
        awesome_tokens = self.base_ids_to_awesome_tokens(base_token_ids)
        concatenated_tokens = "".join(awesome_tokens)

        encoding: tokenizers.Encoding = self.text_tokenizer.encode(concatenated_tokens)
        return encoding.tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        base_token_ids = self.awesome_tokens_to_base_ids(tokens)

        base_tokens = [self.base_tokenizer.vocab[base_token_id] for base_token_id in base_token_ids]
        return self.base_tokenizer.untokenize(tokens=base_tokens)

    def save_tokenizer(self, path: str):
        tokenizer_desc = {
            "base_tokenizer": self.base_tokenizer.name,
            "base_tokenizer_parameters": self.base_tokenizer.parameters,
            "bpe_tokenizer": self.text_tokenizer.to_str(),
        }
        with open(path, "w+") as f:
            json.dump(tokenizer_desc, f)

    @classmethod
    def from_file(cls, path: str) -> "AwesomeMidiTokenizer":
        with open(path, "r") as f:
            tokenizer_desc = json.load(f)

        base_tokenizer_name = tokenizer_desc["base_tokenizer"]
        parameters = tokenizer_desc["base_tokenizer_parameters"]
        bpe_tokenizer_json = tokenizer_desc["bpe_tokenizer"]

        base_tokenizer = generate_base_tokenizer(name=base_tokenizer_name, parameters=parameters)
        tokenizer = Tokenizer.from_str(bpe_tokenizer_json)
        bpe_tokenizer = AwesomeMidiTokenizer(base_tokenizer=base_tokenizer, bpe_tokenizer=tokenizer)

        return bpe_tokenizer
