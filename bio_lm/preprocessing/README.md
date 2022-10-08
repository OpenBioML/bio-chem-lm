# Preprocessing

Before we train a model, we need to train a tokenizer on the corpus.

We train a WordLevel tokenizer on the SELFIES representations. We first process each sequence by splitting the
into a space separated string of SELFIES. We then train a WordLevel tokenizer on the SELFIES representations.

To train a tokenizer from scratch, run

```bash
python build_tokenizer.py
```

The tokenizer has been released to `zpn/pubchem_selfies_world_level_tokenizer` and can be downloaded with