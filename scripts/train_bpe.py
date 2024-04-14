from tokenizers import Regex, Tokenizer, models, trainers, pre_tokenizers


def main():
    tokenizer = Tokenizer(models.BPE())

    # not using a pre-tokenizer turns out to be very inefficient
    # split the tokens into groups before training (concatenate only time tokens)
    velocity_splitter = pre_tokenizers.Split("VELOCITY", behavior="merged_with_next")
    note_on_splitter = pre_tokenizers.Split(Regex("NOTE_ON_.."), behavior="isolated")
    note_off_splitter = pre_tokenizers.Split(Regex("NOTE_OFF_.."), behavior="isolated")

    # in the txt file, new records begin with a newline
    end_line_splitter = pre_tokenizers.Split("\n", behavior="removed")

    # We have to use this - we cannot load saved tokenizer otherwise
    byte_level_pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=False)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [byte_level_pre_tokenizer, end_line_splitter, velocity_splitter, note_on_splitter, note_off_splitter]
    )

    trainer = trainers.BpeTrainer(max_token_length=512, special_tokens=["<CLS>"])
    tokenizer.model = models.BPE()

    print("training...")
    tokenizer.train(["data/maestro-tokenized-one-time.txt"], trainer=trainer)
    print("saving...")
    tokenizer.save("dumps/first_tokenizer.json")


if __name__ == "__main__":
    main()
