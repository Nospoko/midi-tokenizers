import tempfile
from concurrent.futures import ThreadPoolExecutor

import tokenizers
import pandas as pd
from datasets import Dataset, load_dataset
from tokenizers import Regex, Tokenizer, models, trainers, pre_tokenizers

from midi_tokenizers.midi_tokenizer import MidiTokenizer


class BpeTokenizer(MidiTokenizer):
    def __init__(self, base_tokenizer: MidiTokenizer, path: str = None):
        super().__init__()
        self.base_tokenizer = base_tokenizer
        if path is not None:
            self.tokenizer: Tokenizer = Tokenizer.from_file(path=path)
        else:
            # Initialize tokenizer
            self.tokenizer = Tokenizer(model=models.BPE())
            self.tokenizer.pre_tokenizer = self.prepare_text_pre_tokenizer()
            self.tokenizer.model = models.BPE()

            # Train it on maestro
            train_dataset = load_dataset("roszcz/maestro-sustain-v2", split="train")
            self.train(train_dataset=train_dataset)

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
        byte_level_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=False)

        # split the tokens into groups before training (concatenate only time tokens)
        velocity_splitter = pre_tokenizers.Split("VELOCITY", behavior="merged_with_next")
        note_on_splitter = pre_tokenizers.Split(Regex("NOTE_ON_.."), behavior="isolated")
        note_off_splitter = pre_tokenizers.Split(Regex("NOTE_OFF_.."), behavior="isolated")

        # in the txt file, new records begin with a newline
        end_line_splitter = pre_tokenizers.Split("\n", behavior="removed")

        text_pre_tokenizers = [
            byte_level_tokenizer,
            end_line_splitter,
            velocity_splitter,
            note_on_splitter,
            note_off_splitter,
        ]
        return pre_tokenizers.Sequence(text_pre_tokenizers)

    def train(self, train_dataset: Dataset):
        file = tempfile.NamedTemporaryFile()
        self.prepare_data_for_training(file_name=file.name, train_dataset=train_dataset)

        trainer = trainers.BpeTrainer(max_token_length=512, special_tokens=["<CLS>"])
        self.tokenizer.model = models.BPE()
        self.tokenizer.train([file.name], trainer=trainer)

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

    def save_bpe_tokenizer(self, path: str):
        self.tokenizer.save(path=path)
