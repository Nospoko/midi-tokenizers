"""
The midi_tokenizers package provides utilities to tokenize and process MIDI files
for various tasks, including music generation and analysis.

This package includes the following modules and classes:

- MidiTokenizer: Base class for all MIDI tokenizers.
- OneTimeTokenizer: Tokenizer that uses a single time token.
- ExponentialTimeTokenizer: Tokenizer that uses multiple time tokens, rising exponentially.
- QuantizedMidiTokenizer: Tokenizer that uses quantization to first bin the data and then
treats all possible combinations as separate tokens.

Example usage:
    from midi_tokenizers import OneTimeTokenizer

    # Initialize a tokenizer
    tokenizer = OneTimeTokenizer(min_time_unit=0.01, n_velocity_bins=128)

    # Tokenize MIDI notes
    tokens = tokenizer.tokenize(notes)

    # Untokenize to get back MIDI notes
    notes = tokenizer.untokenize(tokens)
"""

from .midi_tokenizer import MidiTokenizer
from .one_time_tokenizer import OneTimeTokenizer
from .quantized_midi_tokenizer import QuantizedMidiTokenizer
from .exponential_time_tokenizer import ExponentialTimeTokenizer

__all__ = [
    "MidiTokenizer",
    "ExponentialTimeTokenizer",
    "OneTimeTokenizer",
    "QuantizedMidiTokenizer",
]
