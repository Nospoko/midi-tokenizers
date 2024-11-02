"""
This module provides a set of MIDI tokenizers that can be used for training and tokenizing MIDI data.

It includes the following classes:
    - AwesomeMidiTokenizer: A MIDI tokenizer that uses BPE and encodes base tokenizer token IDs as characters.
    - MidiTrainableTokenizer: A base class for trainable MIDI tokenizers.
    - BpeMidiTokenizer: A BPE-based tokenizer for MIDI data.

These classes can be used to preprocess and tokenize MIDI data for various applications,
such as training machine learning models.

Example usage:
    from midi_tokenizers import AwesomeMidiTokenizer, BpeMidiTokenizer

    # Initialize a tokenizer
    base_tokenizer = SomeBaseTokenizer()
    tokenizer = BpeMidiTokenizer(base_tokenizer=base_tokenizer)

    # Train the tokenizer
    tokenizer.train(train_dataset)

    # Tokenize MIDI notes
    tokens = tokenizer.tokenize(notes)

    # Save the tokenizer
    tokenizer.save_tokenizer('path/to/save/tokenizer.json')

    # Load the tokenizer
    loaded_tokenizer = BpeMidiTokenizer.from_file('path/to/save/tokenizer.json')
"""

from .bpe_tokenizer import BpeMidiTokenizer
from .trainable_tokenizer import MidiTrainableTokenizer
from .awesome_midi_tokenzier import AwesomeMidiTokenizer

__all__ = [
    "AwesomeMidiTokenizer",
    "MidiTrainableTokenizer",
    "BpeMidiTokenizer",
]
