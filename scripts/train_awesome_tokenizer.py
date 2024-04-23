from datasets import load_dataset, concatenate_datasets

from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer
from midi_trainable_tokenizers.awsome_midi_tokenzier import AwesomeMidiTokenizer

# This is a script for training a BpeMidiTokenizer


def main():
    min_time_unit = 0.01
    n_velocity_bins = 32
    max_token_length = 128
    max_vocab_size = 30000

    base_tokenizer = OneTimeTokenizer(min_time_unit=min_time_unit, n_velocity_bins=n_velocity_bins)
    tokenizer = AwesomeMidiTokenizer(
        base_tokenizer=base_tokenizer, max_vocab_size=max_vocab_size, max_token_length=max_token_length
    )
    basic_dataset = load_dataset("roszcz/maestro-sustain-v2", split="train")
    giant_dataset = load_dataset("roszcz/giant-midi-sustain-v2", split="train")
    dataset = concatenate_datasets([basic_dataset, giant_dataset])

    tokenizer.train(dataset)
    tokenizer.save_tokenizer(f"dumps/big-awesome-{base_tokenizer.name}-{min_time_unit}-{n_velocity_bins}.json")


if __name__ == "__main__":
    main()
