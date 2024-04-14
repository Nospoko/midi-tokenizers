import pandas as pd
from datasets import load_dataset

from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer


def main():
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="train")
    eps = 0.015
    n_velocity_bins = 32

    tokenizer = OneTimeTokenizer(eps=eps, n_velocity_bins=n_velocity_bins)
    filename = "data/maestro-tokenized-one-time.txt"
    with open(filename, "w+") as file:
        for record in dataset:
            notes = pd.DataFrame(record["notes"])
            tokens = tokenizer.tokenize(notes=notes)
            for token in tokens:
                file.write(str(token) + " ")
            file.write("\n")


if __name__ == "__main__":
    main()
