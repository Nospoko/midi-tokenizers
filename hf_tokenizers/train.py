from tokenizers import Tokenizer, models, trainers


def main():
    tokenizer = Tokenizer(models.BPE())

    trainer = trainers.BpeTrainer(max_token_length=10)
    tokenizer.model = models.BPE()
    print("training...")
    tokenizer.train(["data/maestro-tokenized-one-time.txt"], trainer=trainer)
    print("saving...")
    tokenizer.save("dumps/first_tokenizer.json")


if __name__ == "__main__":
    main()
