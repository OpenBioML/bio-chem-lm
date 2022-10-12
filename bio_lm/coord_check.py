import os
from functools import partial

from datasets import load_dataset
from mup import set_base_shapes
from mup.coord_check import get_coord_data, plot_coord_data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from bio_lm.model.config import ElectraConfig
from bio_lm.model.discriminator import ElectraForPreTraining
from bio_lm.options import parse_args
from bio_lm.preprocessing.tokenization import preprocess_fn, tokenize_selfies
from bio_lm.train_utils import load_config


def create_electra(
    model_config,
    base_model_config,
    mup=False,
    readout_zero_init=True,
    query_zero_init=False,
):
    def gen(config, base_config):
        def f():
            loaded = load_config(config)
            loaded["mup"] = mup
            model_config = ElectraConfig(**loaded)
            model = ElectraForPreTraining(model_config)
            if mup is False:
                set_base_shapes(model, None, rescale_params=False)
            else:
                loaded_base = load_config(base_config)
                loaded_base["mup"] = mup
                base_model_config = ElectraConfig(**loaded_base)
                base_model = ElectraForPreTraining(base_model_config)
                set_base_shapes(model, base_model)

            model.apply(
                partial(
                    model._init_weights,
                    readout_zero_init=readout_zero_init,
                    query_zero_init=query_zero_init,
                )
            )

            return model

        return f

    return gen(config=model_config, base_config=base_model_config)


def lazy_model(base_config, model_configs, mup=False):
    r = [128, 1024, 2048, 3072]
    return {
        val: create_electra(
            base_model_config=base_config,
            model_config=c,
            mup=mup,
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
        collate_fn=DataCollatorForLanguageModeling(tokenizer),
    )

    mup = config["mup"]
    optimizer = "adam"
    nseeds = 5
    nsteps = 5
    lr = config["lr"]

    base_config = "model/configs/discriminator/tiny.yaml"

    configs = [
        "model/configs/discriminator/tiny.yaml",
        "model/configs/discriminator/small.yaml",
        "model/configs/discriminator/medium.yaml",
        "model/configs/discriminator/large.yaml",
    ]

    models = lazy_model(
        base_config=base_config,
        model_configs=configs,
        mup=mup,
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
        lossfn="xent",
        cuda=config["gpu"],
        dict_in_out=True,
    )

    prm = "μP" if mup else "SP"

    if lr is None:
        lr = 0.1 if optimizer == "sgd" else 1e-3

    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])

    plot_coord_data(
        df,
        legend="full",
        save_to=f"{config['output_dir']}/{prm.lower()}_electra_{optimizer}_lr{lr}_nseeds{nseeds}_coord.jpg",
        suptitle=f"{prm} electra {optimizer} lr={lr} nseeds={nseeds}",
        face_color="xkcd:light grey" if not mup else None,
    )


if __name__ == "__main__":
    args = parse_args()

    config = {}
    config.update({k: v for k, v in args.__dict__.items() if (v is not None)})

    plot_coords(config)
