import itertools

import pandas as pd
from quantizer_generator import QuantizerGenerator

from midi_tokenizers.midi_tokenizer import MidiTokenizer


class QuantizedMidiTokenizer(MidiTokenizer):
    def __init__(self, quantization_cfg: dict, quantizer_name: str):
        super().__init__()
        quantizer_generator = QuantizerGenerator()
        self.quantizer_name = quantizer_name

        self.quantizer = quantizer_generator.generate_quantizer(
            quantizer_name,
            quantization_cfg=quantization_cfg,
        )

        self.quantization_cfg = quantization_cfg
        self.keys = self.quantizer.keys
        self.specials = ["<CLS>"]

        self.vocab = list(self.specials)

        # add midi tokens to vocab
        self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}
        self.name = "QuantizedMidiTokenizer"

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
        # get product of all possible bin numbers - always use 88 pitch values
        iterables = [range(self.quantization_cfg[f"n_{key}s"]) for key in self.keys if key != "pitch"]
        iterables = [range(21, 109)] + iterables
        src_iterators_product = itertools.product(*iterables)

        for pitch, dstart, duration, velocity in src_iterators_product:
            key = f"{pitch}-{dstart}-{duration}-{velocity}"
            self.vocab.append(key)

    def tokenize(self, notes: pd.DataFrame) -> list[str]:
        tokens = []
        notes = self.quantizer.quantize_frame(df=notes)
        n_samples = len(notes[self.keys[0]])
        for idx in range(n_samples):
            token = "-".join([f"{notes[key][idx]:0.0f}" for key in self.keys])
            tokens.append(token)

        return tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        notes = []
        for token in tokens:
            if token in self.specials:
                continue

            values_txt = token.split("-")
            values = [eval(txt) for txt in values_txt]
            notes.append(values)

        notes = pd.DataFrame(notes, columns=self.keys)
        notes = self.quantizer.apply_quantization(df=notes)

        return notes
