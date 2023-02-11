from tokenizers import Tokenizer, models, trainers 
from transformers import PreTrainedTokenizerFast

def build_aa_tokenizer(save_path):
    # taken from https://github.com/songlab-cal/gpn/blob/af75fb1e15cc943a0f63096c9ba645e968422564/gpn/train_tokenizer_ss.py 
    dataset = ["ACDEFGHIKLMNPQRSTVWY"]
    special_tokens = [
        "[PAD]",
        "[MASK]",
        "[UNK]",
    ]

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(
        vocab_size=len(dataset[0]) + len(special_tokens),
        special_tokens=special_tokens,
        initial_alphabet=list(dataset[0]),
    )
    tokenizer.train_from_iterator(dataset, trainer=trainer, length=len(dataset))

    assert tokenizer.get_vocab_size() > 0 

    tokenizer.save(save_path)


if __name__ == "__main__":
    save_path = "amino_acid_tokenizer.json"
    build_aa_tokenizer(save_path)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=save_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    tokenizer.push_to_hub("zpn/amino_acid_tokenizer")