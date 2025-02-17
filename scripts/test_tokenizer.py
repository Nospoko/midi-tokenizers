import time
from typing import List

import pandas as pd
from fortepyan import MidiPiece
from datasets import load_dataset

from midi_tokenizers.base_tokenizers.exponential_time_tokenizer import ExponentialTimeTokenizer


# TODO typehints
def load_midi_pieces(dataset):
    midi_pieces = []
    for record in dataset:
        piece = MidiPiece.from_huggingface(record=record)
        midi_pieces.append(piece)
    return midi_pieces


# TODO typehints
def measure_tokenization_speed(tokenizer, midi_pieces):
    start_time = time.time()
    total_tokens = 0
    tokenized_pieces = []

    for piece in midi_pieces:
        tokens = tokenizer.tokenize(piece.df)
        total_tokens += len(tokens)
        tokenized_pieces.append((tokens, piece.source))

    end_time = time.time()
    total_time = end_time - start_time
    tokens_per_second = total_tokens / total_time

    print("\nTokenization:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokenization speed: {tokens_per_second:.2f} tokens/second")

    return tokenized_pieces


# TODO typehints
def measure_untokenization_speed(tokenizer, tokenized_pieces):
    start_time = time.time()
    total_tokens = 0
    untokenized_pieces = []

    for tokens, source in tokenized_pieces:
        untokenized_df = tokenizer.untokenize(tokens=tokens)
        total_tokens += len(tokens)
        untokenized_pieces.append(untokenized_df)

    end_time = time.time()
    total_time = end_time - start_time
    tokens_per_second = total_tokens / total_time

    print("\nUntokenization:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Untokenization speed: {tokens_per_second:.2f} tokens/second")

    return untokenized_pieces


def test_tokenizer_accuracy(
    original_pieces: List[MidiPiece],
    untokenized_pieces: List[pd.DataFrame],
    tolerance_ms: float = 10,
) -> dict:
    total_notes = 0
    notes_within_tolerance = 0
    max_error = 0

    for original_piece, untokenized_df in zip(original_pieces, untokenized_pieces):
        original_df = original_piece.df
        original_df = original_df.sort_values("start")
        untokenized_df = untokenized_df.sort_values("start")

        total_notes += len(original_df)

        errors = untokenized_df["start"] - original_df["start"]
        notes_within_tolerance += (errors.abs() < (tolerance_ms / 1000)).sum()
        max_error = max(max_error, errors.abs().max())

    percentage_within_tolerance = (notes_within_tolerance / total_notes) * 100

    return {
        "total_notes": total_notes,
        "notes_within_tolerance": notes_within_tolerance,
        "percentage_within_tolerance": percentage_within_tolerance,
        "max_error_ms": max_error * 1000,
    }


def main():
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="validation")

    print("Loading MidiPieces...")
    loading_start = time.time()
    midi_pieces = load_midi_pieces(dataset)
    print(f"MidiPieces loaded in {time.time() - loading_start:.2f} seconds")

    # Initialize tokenizer with proper config
    tokenizer = ExponentialTimeTokenizer.build_tokenizer(
        tokenizer_config={
            "time_unit": 0.01,
            "max_time_step": 1.0,
            "n_velocity_bins": 32,
            "n_special_ids": 1024,
        }
    )

    # Test serialization/deserialization
    tokenizer_desc = tokenizer.to_dict()
    tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc)

    print(f"\nTesting ExponentialTimeTokenizer on {len(midi_pieces)} records")

    tokenized_pieces = measure_tokenization_speed(
        tokenizer=tokenizer,
        midi_pieces=midi_pieces,
    )

    untokenized_pieces = measure_untokenization_speed(
        tokenizer=tokenizer,
        tokenized_pieces=tokenized_pieces,
    )
    print(midi_pieces[0].df, untokenized_pieces[0])
    print("\nTesting accuracy...")
    results = test_tokenizer_accuracy(midi_pieces, untokenized_pieces)
    print(f"Total notes: {results['total_notes']}")
    print(f"Notes within 10ms: {results['notes_within_tolerance']}")
    print(f"Accuracy: {results['percentage_within_tolerance']:.2f}%")
    print(f"Max error: {results['max_error_ms']:.2f}ms")


if __name__ == "__main__":
    main()
