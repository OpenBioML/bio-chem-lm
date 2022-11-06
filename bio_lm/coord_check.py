import os
from functools import partial

from datasets import load_dataset
from mup import set_base_shapes
from mup.coord_check import get_coord_data, plot_coord_data
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from bio_lm.model.config import ElectraConfig
from bio_lm.model.discriminator import ElectraForPreTraining
from bio_lm.model.generator import ElectraForMaskedLM
from bio_lm.model.electra import Electra
from bio_lm.options import parse_args
from bio_lm.preprocessing.tokenization import preprocess_fn, tokenize_selfies
from bio_lm.train_utils import load_config, tie_weights


BASE = "model/configs/{model_type}/{size}"


def create_electra(
    model_config,
    base_model_config,
    pad_id,
    vocab_size,
    mask_id,
    mup=False,
    readout_zero_init=True,
    query_zero_init=False,
):
    def gen(size, base_size):
        def f():
            disc_config = load_config(
                BASE.format(model_type="discriminator", size=size)
            )
            disc_config["mup"] = mup
            disc_model_config = ElectraConfig(**disc_config)
            discriminator = ElectraForPreTraining(disc_model_config)

            gen_config = load_config(BASE.format(model_type="generator", size=size))
            gen_config["mup"] = mup
            gen_config["vocab_size"] = vocab_size
            gen_model_config = ElectraConfig(**gen_config)
            generator = ElectraForMaskedLM(gen_model_config)

            electra = Electra(
                discriminator=discriminator,
                generator=generator,
                pad_token_id=pad_id,
                mask_token_id=mask_id,
                config=gen_model_config,
            )

            tie_weights(generator, discriminator)

            generator.apply(
                partial(
                    generator._init_weights,
                    readout_zero_init=readout_zero_init,
                    query_zero_init=query_zero_init,
                )
            )

            discriminator.apply(
                partial(
                    discriminator._init_weights,
                    readout_zero_init=readout_zero_init,
                    query_zero_init=query_zero_init,
                )
            )

            if mup is False:
                set_base_shapes(electra, None, rescale_params=False)
            else:
                base_disc_config = load_config(
                    BASE.format(model_type="discriminator", size=base_size)
                )
                base_disc_config["mup"] = mup
                base_disc_model_config = ElectraConfig(**base_disc_config)
                base_discriminator = ElectraForPreTraining(base_disc_model_config)

                base_gen_config = load_config(
                    BASE.format(model_type="generator", size=base_size)
                )
                base_gen_config["mup"] = mup
                base_gen_config["vocab_size"] = vocab_size
                base_gen_model_config = ElectraConfig(**base_gen_config)
                base_generator = ElectraForMaskedLM(base_gen_model_config)

                tie_weights(base_generator, base_discriminator)

                base_generator.apply(
                    partial(
                        generator._init_weights,
                        readout_zero_init=readout_zero_init,
                        query_zero_init=query_zero_init,
                    )
                )

                base_discriminator.apply(
                    partial(
                        discriminator._init_weights,
                        readout_zero_init=readout_zero_init,
                        query_zero_init=query_zero_init,
                    )
                )


                base_electra = Electra(
                    discriminator=base_discriminator,
                    generator=base_generator,
                    pad_token_id=pad_id,
                    mask_token_id=mask_id,
                    config=gen_model_config,
                )


                set_base_shapes(electra, base_electra)

            return electra

        return f

    return gen(size=model_config, base_size=base_model_config)


def lazy_model(base_config, model_configs, pad_id, vocab_size, mask_id, mup=False):
    r = [128, 1024, 2048, 3072]
    return {
        val: create_electra(
            model_config=c,
            base_model_config=base_config,
            mup=mup,
            pad_id=pad_id,
            vocab_size=vocab_size,
            mask_id=mask_id,
        )
        for val, c in zip(r, model_configs)
    }


def plot_coords(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    dataset = load_dataset(config["dataset_name"], split="train", streaming=True)
    dataset = dataset.map(tokenize_selfies, batched=True, batch_size=100)
    dataset = dataset.map(
        lambda x: preprocess_fn(x, tokenizer),
        batched=True,
        remove_columns=[
            "PUBCHEM_COMPOUND_CID",
            "CAN_SELFIES",
            "PUBCHEM_OPENEYE_CAN_SMILES",
            "tokenized",
        ],
    )
    dataset = dataset.with_format("torch")

    dataloader = DataLoader(
        dataset,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False, mlm_probability=0),
    )

    mup = config["mup"]
    optimizer = "adam"
    nseeds = 5
    nsteps = 5
    lr = config["lr"]

    base_config = "tiny.yaml"

    configs = [
        "tiny.yaml",
        "small.yaml",
        "medium.yaml",
        "large.yaml",
    ]

    models = lazy_model(
        base_config=base_config,
        model_configs=configs,
        mup=mup,
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size,
    )

    df = get_coord_data(
        models,
        dataloader,
        mup=mup,
        lr=lr,
        optimizer=optimizer,
        flatten_input=False,
        nseeds=nseeds,
        nsteps=nsteps,
        dict_in_out=True,
        output_name="loss",
        cuda=torch.cuda.is_available(),
    )

    prm = "μP" if mup else "SP"

    if lr is None:
        lr = 0.1 if optimizer == "sgd" else 1e-3

    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])

    plot_coord_data(
        df,
        legend="auto",
        save_to=f"{config['output_dir']}/{prm.lower()}_electra_model_{optimizer}_lr{lr}_nseeds{nseeds}_coord.jpg",
        suptitle=f"{prm} electra model {optimizer} lr={lr} nseeds={nseeds}",
        face_color="xkcd:light grey" if not mup else None,
        name_not_contains="out[",
        name_contains="electra"
    )


if __name__ == "__main__":
    args = parse_args()

    config = {}
    config.update({k: v for k, v in args.__dict__.items() if (v is not None)})

    plot_coords(config)
