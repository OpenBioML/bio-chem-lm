import numpy as np
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
from random import choice

from bio_lm.preprocessing.tokenization import preprocess_fn, tokenize_selfies
from bio_lm.train_utils import standardize


def get_mean_std(dataset):
    return np.mean(dataset["target"]), np.std(dataset["target"])


def get_class_weights(dataset):
    return compute_class_weight(
        class_weight="balanced",
        classes=np.unique(dataset["target"]),
        y=dataset["target"],
    )


def get_statistics(dataset):
    if hasattr(dataset.features["target"], "num_classes"):
        problem_type = "classification"
        num_labels = dataset.features["target"].num_classes
        class_weights = get_class_weights(dataset)
        mean, std = None, None
    else:
        problem_type = "regression"
        num_labels = 1
        class_weights = None
        mean, std = get_mean_std(dataset)

    return problem_type, num_labels, class_weights, mean, std


def get_training_statistics(config):
    dataset = load_dataset(config["dataset_name"], split="train")

    return get_statistics(dataset)


def load_finetune_dataset(config, tokenizer, split="train"):
    dataset = load_dataset(config["dataset_name"], split=split)

    problem_type, num_labels, class_weights, mean, std = get_training_statistics(config)

    dataset = dataset.shuffle(seed=config["seed"])
    dataset = dataset.map(
        lambda x: tokenize_selfies(x, col_name="selfies"),
        batched=True,
        batch_size=config[f"{split}_batch_size"],
    )
    dataset = dataset.map(
        lambda x: preprocess_fn(x, tokenizer),
        batched=True,
        remove_columns=[
            "smiles",
            "selfies",
            "tokenized",
        ],
    )

    if mean is not None or std is not None:
        dataset = dataset.map(
            lambda x: standardize(x, mean=mean, std=std),
            batched=True,
        )

    dataset = dataset.with_format("torch")

    dataloader = DataLoader(
        dataset,
        collate_fn=DefaultDataCollator(),
        batch_size=config[f"{split}_batch_size"],
    )

    return dataloader, problem_type, num_labels, class_weights



def impute_aa(examples):
    vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'] 
    mapping = {"X": vocab, "B": ["D", "N"], "J": ["I", "L"], "Z": ["E", "G"]}
    imputed = []

    for seq in examples["sequence"]:
        updated_seq = seq
        for i, aa in enumerate(seq):
            if aa in mapping:
                changed_aa = choice(mapping[aa])
                updated_seq = updated_seq[:i] + changed_aa + updated_seq[i+1: ]

        imputed.append(updated_seq)

    examples["imputed"] = imputed

    return examples
            
    