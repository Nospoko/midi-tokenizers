import time

from fortepyan import MidiPiece
from datasets import load_dataset

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

    for tokens, source in tokenized_pieces:
        tokenizer.untokenize(tokens=tokens)
        total_tokens += len(tokens)

    end_time = time.time()
    total_time = end_time - start_time
    tokens_per_second = total_tokens / total_time

    return total_time, total_tokens, tokens_per_second


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

    print("\nRunning speed test for ExponentialTimeTokenizer tokenizer on validation split")
    print(f"Dataset size: {len(midi_pieces)} records")

    # Measure tokenization speed
    tokenization_time, total_tokens, tokenization_speed, tokenized_pieces = measure_tokenization_speed(
        tokenizer,
        midi_pieces,
    )
    print("\nTokenization:")
    print(f"Total time: {tokenization_time:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokenization speed: {tokenization_speed:.2f} tokens/second")

    # Measure untokenization speed
    untokenization_time, _, untokenization_speed = measure_untokenization_speed(
        tokenizer,
        tokenized_pieces,
    )
    print("\nUntokenization:")
    print(f"Total time: {untokenization_time:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Untokenization speed: {untokenization_speed:.2f} tokens/second")


if __name__ == "__main__":
    main()
