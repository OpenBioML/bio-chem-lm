import getpass
from functools import partial

from accelerate import Accelerator
import torch
from datasets import load_dataset
from mup import MuAdam, set_base_shapes
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
import wandb

from bio_lm.metrics import MetricDict, format_metrics
from bio_lm.model.config import ElectraConfig
from bio_lm.model.discriminator import ElectraForPreTraining
from bio_lm.model.electra import Electra
from bio_lm.model.generator import ElectraForMaskedLM
from bio_lm.options import parse_args
from bio_lm.preprocessing.tokenization import preprocess_fn, tokenize_selfies
from bio_lm.train_utils import load_config, make_shapes, tie_weights


def load_data(config, tokenizer, split="train"):
    dataset = load_dataset(config["dataset_name"], split=split, streaming=True)
    dataset = dataset.map(
        tokenize_selfies, batched=True, batch_size=config[f"{split}_batch_size"]
    )
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
        batch_size=config[f"{split}_batch_size"],
    )

    return dataloader


def train(config):
    accelerator = Accelerator(log_with="wandb")

    name = config["wandb_exp_name"] if "wandb_exp_name" in config else None
    accelerator.init_trackers(
        project_name=config["wandb_project"],
        config=config,
        init_kwargs={"wandb": {"entity": config["wandb_entity"], "name": name}},
    )

    if config["wandb"]:
        if config["disc_base_shapes"] is None:
            config["disc_base_shapes"] = f"{wandb.run.dir}/disc_base_shapes"
        if config["gen_base_shapes"] is None:
            config["gen_base_shapes"] = f"{wandb.run.dir}/gen_base_shapes"

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    train_dataloader = load_data(config, tokenizer)

    val_dataloader = load_data(config, tokenizer, split="validation")

    if config["mup"]:
        # generate the base shapes file
        disc_filename = make_shapes(
            config["discriminator_base_config"],
            delta_model_config="model/configs/discriminator/small.yaml",
            save_dir=config["disc_base_shapes"],
            generator=False,
        )

        gen_filename = make_shapes(
            config["generator_base_config"],
            delta_model_config="model/configs/generator/small.yaml",
            save_dir=config["gen_base_shapes"],
            generator=True,
        )

        disc_config = load_config(config["discriminator_config"])
        disc_config["mup"] = True
        disc_config["vocab_size"] = tokenizer.vocab_size
        discriminator_config = ElectraConfig(**disc_config)
        discriminator = ElectraForPreTraining(discriminator_config)

        set_base_shapes(discriminator, disc_filename)

        gen_config = load_config(config["generator_config"])
        gen_config["mup"] = True
        gen_config["vocab_size"] = tokenizer.vocab_size
        generator_config = ElectraConfig(**gen_config)
        generator = ElectraForMaskedLM(generator_config)

        set_base_shapes(generator, gen_filename)

    else:
        disc_config = load_config(config["discriminator_config"])
        discriminator_config = ElectraConfig(disc_config)
        discriminator = ElectraForPreTraining(discriminator_config)

        gen_config = load_config(config["generator_config"])
        generator_config = ElectraConfig(**gen_config)
        generator = ElectraForMaskedLM(generator_config)

    generator.apply(
        partial(
            generator._init_weights,
            readout_zero_init=generator_config.readout_zero_init,
            query_zero_init=config["query_zero_init"],
        )
    )

    discriminator.apply(
        partial(
            discriminator._init_weights,
            readout_zero_init=discriminator_config.readout_zero_init,
            query_zero_init=config["query_zero_init"],
        )
    )

    device = accelerator.device

    generator.to(device)
    discriminator.to(device)

    tie_weights(generator, discriminator)

    model = Electra(
        discriminator=discriminator,
        generator=generator,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        config=discriminator_config,
    )

    model.to(device)

    if config["mup"]:
        optimizer = MuAdam(model.parameters(), lr=config["lr"])
    else:
        optimizer = Adam(model.parameters(), lr=config["lr"])

    if config["scheduler"]:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["num_warmup_steps"],
            num_training_steps=config["num_training_steps"],
        )

    train_metrics = MetricDict(
        [
            "loss",
            "mlm_loss",
            "disc_loss",
            "gen_acc",
            "disc_acc",
        ],
        device=device,
    )
    val_metrics = MetricDict(
        [
            "loss",
            "mlm_loss",
            "disc_loss",
            "gen_acc",
            "disc_acc",
        ],
        device=device,
    )

    if config["scheduler"]:
        model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )
    else:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader 
        )

    train_dataloader = iter(train_dataloader)

    for epoch in range(config["num_epochs"]):
        for step in tqdm(range(config["num_steps_per_epoch"]), desc="Training"):
            if step == 5 and config["debug"]:
                break

            optimizer.zero_grad()
            batch = next(train_dataloader)
            model.train()

            loss = model(**batch)
            accelerator.backward(loss["loss"])

            if config["global_clip_norm"] and config["global_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["global_clip_norm"],
                    error_if_nonfinite=True,
                )

            optimizer.step()

            if config["scheduler"]:
                scheduler.step()

            train_metrics.update(
                {key: value.detach().cpu().numpy() for key, value in loss.items()}
            )

        with torch.no_grad():
            # we only evalute with 1000 steps since there are 10M data points!!
            for i, batch in enumerate(
                tqdm(
                    val_dataloader,
                    desc="Validation",
                    total=config["num_steps_per_epoch"],
                )
            ):
                if i == 5 and config["debug"]:
                    break

                if i == config["num_steps_per_epoch"]:
                    break

                model.eval()
                loss = model(**batch)

                val_metrics.update(
                    {key: value.detach().cpu().numpy() for key, value in loss.items()}
                )

        log_train = {
            f"train_{key}": value for key, value in train_metrics.compute().items()
        }
        log_val = {f"val_{key}": value for key, value in val_metrics.compute().items()}

        if config["wandb"]:
            accelerator.log({**log_train, **log_val})

        print(format_metrics(log_train, "train", f" epoch {epoch} "))
        print(format_metrics(log_val, "val", f" epoch {epoch} "))

        train_metrics.reset_metrics()
        val_metrics.reset_metrics()


if __name__ == "__main__":
    args = parse_args()

    config = {}
    config.update({k: v for k, v in args.__dict__.items()})

    if config["wandb"] or config["wandb_entity"]:
        # if we set wandb_entity, we set to True automatically
        config["wandb"] = True
        config["wandb_entity"] = (
            config["wandb_entity"] if config["wandb_entity"] else getpass.getuser()
        )

    print("| configs: ")
    for k, v in config.items():
        print("  |", k, " : ", v)

    train(config=config)
