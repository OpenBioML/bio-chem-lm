import os
from functools import partial

from datasets import load_dataset
from mup import set_base_shapes
from mup.coord_check import get_coord_data, plot_coord_data
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from bio_lm.model.deberta.config import DebertaV2Config
from bio_lm.model.deberta.discriminator import DebertaV2ForPreTraining
from bio_lm.model.deberta.generator import DebertaV2ForMaskedLM

from bio_lm.model.electra.configuring_electra import ElectraConfig
from bio_lm.model.electra.modeling_electra import ElectraForPreTraining, ElectraForMaskedLM, Electra
from bio_lm.options import parse_args
from bio_lm.preprocessing.tokenization import preprocess_fn, tokenize_selfies
from bio_lm.train_utils import load_config, tie_weights


BASE = "model/configs/{arch_name}/{model_type}/{size}"


def create_model(
    model_config,
    arch_name,
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
            config_base_class = ElectraConfig if arch_name == "electra" else DebertaV2Config
            disc_base_class = ElectraForPreTraining if arch_name == "electra" else DebertaV2ForPreTraining 
            gen_base_class = ElectraForMaskedLM if arch_name == "electra" else DebertaV2ForMaskedLM 
            disc_config = load_config(
                BASE.format(arch_name=arch_name, model_type="discriminator", size=size)
            )
            disc_config["mup"] = mup
            disc_config["vocab_size"] = vocab_size
            disc_model_config = config_base_class(**disc_config)
            discriminator = disc_base_class(disc_model_config)

            gen_config = load_config(BASE.format(arch_name=arch_name, model_type="generator", size=size))
            gen_config["mup"] = mup
            gen_config["vocab_size"] = vocab_size
            gen_model_config = config_base_class(**gen_config)
            generator = gen_base_class(gen_model_config)

            if config["arch_name"] == "electra":
                tie_weights(generator, discriminator)

            electra = Electra(
                discriminator=discriminator,
                generator=generator,
                pad_token_id=pad_id,
                mask_token_id=mask_id,
                config=gen_model_config,
            )

            if mup is False:
                set_base_shapes(electra, None)
            else:
                base_disc_config = load_config(
                    BASE.format(arch_name=arch_name, model_type="discriminator", size=base_size)
                )
                base_disc_config["mup"] = mup
                base_disc_config["vocab_size"] = vocab_size
                base_disc_model_config = config_base_class(**base_disc_config)
                base_discriminator = disc_base_class(base_disc_model_config)

                base_gen_config = load_config(
                    BASE.format(arch_name=arch_name, model_type="generator", size=base_size)
                )
                base_gen_config["mup"] = mup
                base_gen_config["vocab_size"] = vocab_size
                base_gen_model_config = config_base_class(**base_gen_config)
                base_generator = gen_base_class(base_gen_model_config)

                if config["arch_name"] == "electra":
                    tie_weights(base_generator, base_discriminator)

                base_electra = Electra(
                    discriminator=base_discriminator,
                    generator=base_generator,
                    pad_token_id=pad_id,
                    mask_token_id=mask_id,
                    config=gen_model_config,
                )

                set_base_shapes(electra, base_electra)

            electra.apply(
                partial(
                    electra._init_weights,
                    readout_zero_init=readout_zero_init,
                    query_zero_init=query_zero_init,
                )
            )

            return electra

        return f

    return gen(size=model_config, base_size=base_model_config)


def lazy_model(base_config, model_configs, arch_name, pad_id, vocab_size, mask_id, query_zero_init=False, mup=False):
    r = [128, 1024, 2048, 3072]
    return {
        val: create_model(
            model_config=c,
            arch_name=arch_name,
            base_model_config=base_config,
            mup=mup,
            pad_id=pad_id,
            vocab_size=vocab_size,
            mask_id=mask_id,
            query_zero_init=query_zero_init
        )
        for val, c in zip(r, model_configs)
    }


def plot_coords(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    dataset = load_dataset(config["dataset_name"], split="train", streaming=True)
    col_name = "CAN_SELFIES" if config["dataset_name"] == "zpn/pubchem_selfies" else "selfies"
    remove_cols = [
            "PUBCHEM_COMPOUND_CID",
            "CAN_SELFIES",
            "PUBCHEM_OPENEYE_CAN_SMILES",
            "tokenized",
        ] if config["dataset_name"] == "zpn/pubchem_selfies" else ["selfies", "tokenized", "smiles", "id"]
    dataset = dataset.map(lambda x: tokenize_selfies(x, col_name=col_name), batched=True, batch_size=100)
    dataset = dataset.map(
        lambda x: preprocess_fn(x, tokenizer),
        batched=True,
        remove_columns=remove_cols
    )
    dataset = dataset.with_format("torch")

    dataloader = DataLoader(
        dataset, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False, mlm_probability=0)
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
        arch_name=config["arch_name"],
        mup=mup,
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size,
        query_zero_init=config["query_zero_init"],
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
        filter_trainable_by_name=lambda x: True if ":out[" not in x and x != "" else False
    )

    prm = "μP" if mup else "SP"

    if lr is None:
        lr = 0.1 if optimizer == "sgd" else 1e-3

    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])

    plot_coord_data(
        df,
        legend="auto",
        save_to=f"{config['output_dir']}/{prm.lower()}_{config['arch_name']}_model_{optimizer}_lr{lr}_nseeds{nseeds}_coord.jpg",
        suptitle=f"{prm} {config['arch_name']} model {optimizer} lr={lr} nseeds={nseeds}",
        face_color="xkcd:light grey" if not mup else None,
        name_not_contains="out[",
        name_contains="generator",
    )


if __name__ == "__main__":
    args = parse_args()

    config = {}
    config.update({k: v for k, v in args.__dict__.items() if (v is not None)})

    plot_coords(config)
