from tokenizers import Tokenizer, models, pre_tokenizers


def main():
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # trainer = trainers.BpeTrainer(special_tokens=["<|endoftext|>"])
    tokenizer.model = models.BPE()

    tokenizer.train(["data/maestro-tokenized-one-time.txt"])
    tokenizer.save("dumps/first_tokenizer.json")


if __name__ == "__main__":
    main()
