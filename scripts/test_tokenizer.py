import time
from typing import List

import pandas as pd
from fortepyan import MidiPiece
from datasets import load_dataset

from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers.no_loss_tokenizer import ExponentialTimeTokenizer


def load_midi_pieces(dataset):
    midi_pieces = []
    for record in dataset:
        piece = MidiPiece.from_huggingface(record=record)
        midi_pieces.append(piece)
    return midi_pieces


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

    return total_time, total_tokens, tokens_per_second, tokenized_pieces


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

    return total_time, total_tokens, tokens_per_second, untokenized_pieces


def test_tokenizer_accuracy(
    original_pieces: List[MidiPiece], untokenized_pieces: List[pd.DataFrame], tolerance_ms: float = 10
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
    # Load the dataset
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="validation")

    print("Loading MidiPieces...")
    loading_start_time = time.time()
    midi_pieces = load_midi_pieces(dataset)
    loading_end_time = time.time()
    loading_time = loading_end_time - loading_start_time
    print(f"MidiPieces loaded in {loading_time:.2f} seconds")

    tokenizer = ExponentialTimeTokenizer(min_time_unit=0.01, n_velocity_bins=32)
    # tokenizer = AwesomeMidiTokenizer.from_file("dumps/awesome_tokenizers/awesome-tokenizer-test-2024-06-11_17-11-44.json")

    print("\nRunning speed and accuracy test for ExponentialTimeTokenizer on validation split")
    print(f"Dataset size: {len(midi_pieces)} records")

    # Measure tokenization speed
    tokenization_time, total_tokens, tokenization_speed, tokenized_pieces = measure_tokenization_speed(tokenizer, midi_pieces)
    print("\nTokenization:")
    print(f"Total time: {tokenization_time:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokenization speed: {tokenization_speed:.2f} tokens/second")

    # Measure untokenization speed
    untokenization_time, _, untokenization_speed, untokenized_pieces = measure_untokenization_speed(tokenizer, tokenized_pieces)
    print("\nUntokenization:")
    print(f"Total time: {untokenization_time:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Untokenization speed: {untokenization_speed:.2f} tokens/second")

    # Test accuracy
    print("\nTesting accuracy...")
    accuracy_results = test_tokenizer_accuracy(midi_pieces, untokenized_pieces)
    print(f"Total notes: {accuracy_results['total_notes']}")
    print(f"Notes within 10ms tolerance: {accuracy_results['notes_within_tolerance']}")
    print(f"Percentage within tolerance: {accuracy_results['percentage_within_tolerance']:.2f}%")
    print(f"Max absolute error: {accuracy_results['max_error_ms']:.2f}ms")


if __name__ == "__main__":
    main()
