import selfies as sf


def tokenize_selfies(examples):
    tokenized_examples = []

    selfies = examples["CAN_SELFIES"]
    for example in selfies:
        all_tokens = list(sf.split_selfies(example))

        # to encode with tokenizers, input needs to be a string
        # easier to extract tokens and add whitespace then split in the tokenizer
        tokenized_str = " ".join(all_tokens)
        tokenized_examples.append(tokenized_str)

    return {"tokenized": tokenized_examples}


def preprocess_fn(examples, tokenizer):
    result = tokenizer(examples["tokenized"], padding="max_length", truncation=True)
    return result


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("zpn/pubchem_selfies")

    train_ds = dataset["train"]

    item = train_ds[0]
    #print(item)
    #print(tokenize_selfies({"CAN_SELFIES": [item["CAN_SELFIES"]]}))
