import multiprocessing as mp

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from bio_lm.preprocessing.tokenization import tokenize_selfies

# this is max length of the longest sequence in the dataset is 1773
# setting to 512 for now, 86,262 / 89,000,000 are longer than 512 tokens
MAX_LENGTH = 512


def train_tokenizer(
    save_path, dataset_name, batch_size=10_000, num_workers=mp.cpu_count()
):
    dataset = load_dataset(dataset_name, num_proc=num_workers)

    # only tokenize training data
    train_ds = dataset["train"]

    train_ds = train_ds.map(lambda x: tokenize_selfies(x, col_name="selfies"), batched=True, num_proc=num_workers)

    def batch_loader(batch_size=batch_size):
        for i in range(0, len(train_ds), batch_size):
            yield train_ds[i : i + batch_size]["tokenized"]

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    # equivalent to the `.split()` method
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    trainer = trainers.WordLevelTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    tokenizer.train_from_iterator(batch_loader(), trainer=trainer, length=len(train_ds))
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    assert tokenizer.get_vocab_size() > 0, "Vocab size is 0, something went wrong"

    tokenizer.save(save_path)


if __name__ == "__main__":
    save_path = "zinc20_tokenizer.json"
    train_tokenizer(save_path, "zpn/zinc20")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=save_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=MAX_LENGTH,
    )

    tokenizer.push_to_hub("zpn/zinc20_wordlevel_dissociation")
