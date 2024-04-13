from abc import abstractmethod

import pandas as pd


class MidiTokenizer:
    def __init__(self):
        self.token_to_id = None

    @abstractmethod
    def tokenize(self, record: dict) -> list[str]:
        raise NotImplementedError("Your encoder needs *tokenize* implementation")

    @abstractmethod
    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Your encoder needs *untokenize* implementation")

    def decode(self, token_ids: list[int]) -> pd.DataFrame:
        tokens = [self.vocab[token_id] for token_id in token_ids]
        df = self.untokenize(tokens)

        return df

    def encode(self, record: dict) -> list[int]:
        tokens = self.tokenize(record)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids
