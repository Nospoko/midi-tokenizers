import itertools

import pandas as pd

from midi_tokenizers.midi_tokenizer import MidiTokenizer
from midi_tokenizers_generation.quantizer_generator import generate_quantizer


class QuantizedMidiTokenizer(MidiTokenizer):
    """
    Tokenizer for quantized MIDI data.

    Inherits from MidiTokenizer and utilizes a quantizer to process MIDI data.

    Attributes:
        quantization_cfg (dict): Configuration for quantization.
        quantizer_name (str): Name of the quantizer.
        special_tokens (list[str]): List of special tokens.
        quantizer: Quantizer object generated based on the configuration.
        keys (list[str]): Keys used for quantization.
        vocab (list[str]): Vocabulary of tokens.
        token_to_id (dict): Mapping from tokens to their IDs.
        name (str): Name of the tokenizer.
    """

    def __init__(
        self,
        quantization_cfg: dict,
        quantizer_name: str,
        special_tokens: list[str] = None,
    ):
        """
        Initialize the QuantizedMidiTokenizer with specified quantization configuration and quantizer name.

        Parameters:
        quantization_cfg (dict): Configuration for quantization.
        quantizer_name (str): Name of the quantizer.
        special_tokens (list[str], optional): List of special tokens. Defaults to None.
        """
        super().__init__(special_tokens=special_tokens)
        self.quantizer_name = quantizer_name

        self.quantizer = generate_quantizer(
            quantizer_name,
            quantization_cfg=quantization_cfg,
        )

        self.quantization_cfg = quantization_cfg
        self.keys = self.quantizer.keys

        self.vocab = list(self.special_tokens)

        # add midi tokens to vocab
        self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}
        self.name = "QuantizedMidiTokenizer"
        self.pad_token_id = self.token_to_id["<PAD>"]

    def __rich_repr__(self):
        yield "QuantizedMidiTokenizer"
        yield "vocab_size", self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def parameters(self):
        return {"quantization_cfg": self.quantization_cfg, "quantizer_name": self.quantizer_name}

    def _build_vocab(self):
        """
        Build the vocabulary of the QuantizedMidiTokenizer,
        including all possible combinations of quantized pitch, dstart, duration, and velocity values.
        """
        # get product of all possible bin numbers - always use 88 pitch values
        iterables = [range(self.quantization_cfg[f"n_{key}s"]) for key in self.keys if key != "pitch"]
        iterables = [range(21, 109)] + iterables
        src_iterators_product = itertools.product(*iterables)

        for pitch, dstart, duration, velocity in src_iterators_product:
            key = f"{pitch}-{dstart}-{duration}-{velocity}"
            self.vocab.append(key)

    def tokenize(self, notes: pd.DataFrame) -> list[str]:
        """
        Convert a DataFrame of MIDI notes into a list of tokens.

        Parameters:
        notes (pd.DataFrame): The DataFrame of MIDI notes to tokenize.

        Returns:
        list[str]: The list of tokens.
        """
        tokens = []
        notes = self.quantizer.quantize_frame(df=notes)
        n_samples = len(notes[self.keys[0]])
        for idx in range(n_samples):
            token = "-".join([f"{notes[key][idx]:0.0f}" for key in self.keys])
            tokens.append(token)

        return tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        """
        Convert a list of tokens back into a DataFrame of MIDI notes.

        Parameters:
        tokens (list[str]): The list of tokens to untokenize.

        Returns:
        pd.DataFrame: The DataFrame of untokenized MIDI notes.
        """
        notes = []
        for token in tokens:
            if token in self.special_tokens:
                continue

            values_txt = token.split("-")
            values = [eval(txt) for txt in values_txt]
            notes.append(values)

        notes = pd.DataFrame(notes, columns=self.keys)
        notes = self.quantizer.apply_quantization(df=notes)

        return notes
