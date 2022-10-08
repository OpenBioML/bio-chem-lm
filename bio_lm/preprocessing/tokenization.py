def tokenize_selfies(examples):
    tokenized_examples = []

    selfies = examples["CAN_SELFIES"]
    for example in selfies:
        location = example.find("]")
        all_tokens = []
        while location >= 0:
            all_tokens.append(example[0 : location + 1])
            example = example[location + 1 :]
            location = example.find("]")

        # to encode with tokenizers, input needs to be a string
        # easier to extract tokens and add whitespace then split in the tokenizer
        tokenized_str = " ".join(all_tokens)
        tokenized_examples.append(tokenized_str)

    return {"tokenized": tokenized_examples}


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("zpn/pubchem_selfies")

    train_ds = dataset["train"]

    item = train_ds[0]
    print(tokenize_selfies(item["CAN_SELFIES"]))
