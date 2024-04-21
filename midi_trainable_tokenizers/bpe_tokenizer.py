import json
from concurrent.futures import ThreadPoolExecutor

import tokenizers
import pandas as pd
from datasets import Dataset
from tokenizers import Regex, Tokenizer, models, trainers, pre_tokenizers

from midi_tokenizers.midi_tokenizer import MidiTokenizer
from midi_trainable_tokenizers.trainable_tokenizer import MidiTrainableTokenizer
from object_generators.base_tokenizer_generator import generate_tokenizer as generate_base_tokenizer


class BpeMidiTokenizer(MidiTrainableTokenizer):
    def __init__(self, base_tokenizer: MidiTokenizer, bpe_tokenizer: Tokenizer = None, max_vocab_size: int = None):
        super().__init__()
        self.base_tokenizer = base_tokenizer
        self.tokenizer = bpe_tokenizer
        self.name = "BpeTokenizer"
        self.max_vocab_size = max_vocab_size

        if self.max_vocab_size is None:
            self.max_vocab_size = 30000  # default BpeTrainer vocab_size

        if self.tokenizer is None:
            # Initialize empty tokenizer and a trainer
            self.tokenizer = Tokenizer(model=models.BPE())
            self.tokenizer.pre_tokenizer = self.prepare_text_pre_tokenizer()
            self.tokenizer.model = models.BPE()
            self.trainer = trainers.BpeTrainer(
                vocab_size=self.max_vocab_size,
                max_token_length=512,
                special_tokens=["<CLS>"],
            )

        self.vocab = self.tokenizer.get_vocab()

    def prepare_data_for_training(self, file_name: str, train_dataset: Dataset):
        def process_record(record):
            notes = pd.DataFrame(record["notes"])
            tokens = self.base_tokenizer.tokenize(notes=notes)
            return " ".join(str(token) for token in tokens) + "\n"

        with open(file=file_name, mode="w+") as file, ThreadPoolExecutor() as executor:
            # Process records concurrently
            for result in executor.map(process_record, train_dataset):
                file.write(result)

    def prepare_text_pre_tokenizer(self):
        # We have to use this - we cannot load saved tokenizer otherwise
        byte_level_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)

        # split the tokens into groups before training (concatenate only time tokens and velocity+note_on tokens)
        note_on_splitter = pre_tokenizers.Split(Regex("VELOCITY_..? NOTE_ON_.."), behavior="isolated")
        note_off_splitter = pre_tokenizers.Split(Regex("VELOCITY_..? NOTE_OFF_.."), behavior="isolated")

        # in the txt file, new records begin with a newline
        end_line_splitter = pre_tokenizers.Split("\n", behavior="removed")

        text_pre_tokenizers = [
            end_line_splitter,
            note_on_splitter,
            note_off_splitter,
            byte_level_tokenizer,
        ]
        return pre_tokenizers.Sequence(text_pre_tokenizers)

    def tokenize(self, notes: pd.DataFrame) -> list[str]:
        tokens = self.base_tokenizer.tokenize(notes)
        concatenated_tokens = " ".join(tokens)

        encoding: tokenizers.Encoding = self.tokenizer.encode(concatenated_tokens)
        return encoding.tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        concatenated_tokens = "".join(tokens)
        # 288 is  unicode of Ġ letter - which is special letter in ByteLevel pre-tokenizer...
        # (yes, it has to be that complicated)
        concatenated_tokens = concatenated_tokens.replace(chr(288), " ")
        new_tokens = concatenated_tokens.split(" ")
        return self.base_tokenizer.untokenize(tokens=new_tokens)

    def save_tokenizer(self, path: str):
        tokenizer_desc = {
            "base_tokenizer": self.base_tokenizer.name,
            "base_tokenizer_parameters": self.base_tokenizer.parameters,
            "bpe_tokenizer": self.tokenizer.to_str(),
        }
        with open(path, "w+") as f:
            json.dump(tokenizer_desc, f)

    @classmethod
    def from_file(cls, path: str) -> "BpeMidiTokenizer":
        with open(path, "r") as f:
            tokenizer_desc = json.load(f)

        base_tokenizer_name = tokenizer_desc["base_tokenizer"]
        parameters = tokenizer_desc["base_tokenizer_parameters"]
        bpe_tokenizer_json = tokenizer_desc["bpe_tokenizer"]

        base_tokenizer = generate_base_tokenizer(name=base_tokenizer_name, parameters=parameters)
        tokenizer = Tokenizer.from_str(bpe_tokenizer_json)
        bpe_tokenizer = BpeMidiTokenizer(base_tokenizer=base_tokenizer, bpe_tokenizer=tokenizer)

        return bpe_tokenizer
