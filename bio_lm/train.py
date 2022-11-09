import getpass
import os
from functools import partial

import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from datasets import load_dataset
from mup import MuAdam, set_base_shapes
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

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
    dataset = dataset.shuffle(seed=config["seed"], buffer_size=10_000)
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
        # we set mlm to false since we do MLM in the `Electra` model
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer, mlm=False, mlm_probability=0
        ),
        batch_size=config[f"{split}_batch_size"],
    )

    return dataloader


def train(accelerator, config):
    set_seed(config["seed"])
    accelerator.print(f"NUM GPUS: {accelerator.num_processes}")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    with accelerator.main_process_first():
        train_dataloader = load_data(config, tokenizer)
        val_dataloader = load_data(config, tokenizer, split="validation")

    if config["mup"]:
        # generate the base shapes file
        electra_shapes_filename = make_shapes(
            base_size=config["base_config_size"],
            delta_size="small.yaml",
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_token_id,
            mask_id=tokenizer.mask_token_id,
            save_dir=config["base_shapes_dir"],
        )

        disc_config = load_config(config["discriminator_config"])
        disc_config["mup"] = True
        disc_config["vocab_size"] = tokenizer.vocab_size
        discriminator_config = ElectraConfig(**disc_config)
        discriminator = ElectraForPreTraining(discriminator_config)

        gen_config = load_config(config["generator_config"])
        gen_config["mup"] = True
        gen_config["vocab_size"] = tokenizer.vocab_size
        generator_config = ElectraConfig(**gen_config)
        generator = ElectraForMaskedLM(generator_config)

        tie_weights(generator, discriminator)

        model = Electra(
            discriminator=discriminator,
            generator=generator,
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            config=discriminator_config,
        )

        set_base_shapes(model, electra_shapes_filename)

    else:
        disc_config = load_config(config["discriminator_config"])
        discriminator_config = ElectraConfig(disc_config)
        discriminator = ElectraForPreTraining(discriminator_config)

        gen_config = load_config(config["generator_config"])
        generator_config = ElectraConfig(**gen_config)
        generator = ElectraForMaskedLM(generator_config)

        tie_weights(generator, discriminator)

        model = Electra(
            discriminator=discriminator,
            generator=generator,
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            config=discriminator_config,
        )

    model.apply(
        partial(
            model._init_weights,
            readout_zero_init=config["readout_zero_init"],
            query_zero_init=config["query_zero_init"],
        )
    )

    device = accelerator.device
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
        (
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            scheduler,
        ) = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )
    else:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )

    train_dataloader = iter(train_dataloader)

    for epoch in range(config["num_epochs"]):
        for step in tqdm(
            range(config["num_steps_per_epoch"]),
            desc="Training",
            disable=not accelerator.is_local_main_process,
        ):
            if step == 5 and config["debug"]:
                break

            optimizer.zero_grad()
            batch = next(train_dataloader)
            model.train()

            loss = model(**batch)
            accelerator.backward(loss["loss"])

            if config["global_clip_norm"] and config["global_clip_norm"] > 0:
                if accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(), config["global_clip_norm"]
                        )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config["global_clip_norm"],
                        error_if_nonfinite=True,
                    )

            optimizer.step()

            if config["scheduler"]:
                scheduler.step()

            loss = {key: value.detach() for key, value in loss.items()}
            loss = accelerator.gather_for_metrics(loss)

            train_metrics.update(loss)

        with torch.no_grad():
            # we only evalute with N steps since there are 10M data points!!
            for i, batch in enumerate(
                tqdm(
                    val_dataloader,
                    desc="Validation",
                    total=config["num_eval_steps"],
                    disable=not accelerator.is_local_main_process,
                )
            ):
                if i == 5 and config["debug"]:
                    break

                if i == config["num_eval_steps"]:
                    break

                model.eval()
                loss = model(**batch)

                loss = {key: value.detach() for key, value in loss.items()}
                loss = accelerator.gather_for_metrics(loss)

                val_metrics.update(loss)

        log_train = {
            f"train_{key}": value for key, value in train_metrics.compute().items()
        }
        log_val = {f"val_{key}": value for key, value in val_metrics.compute().items()}

        if config["wandb"]:
            accelerator.log({**log_train, **log_val})

        accelerator.print(format_metrics(log_train, "train", f" epoch {epoch} "))
        accelerator.print(format_metrics(log_val, "val", f" epoch {epoch} "))

        train_metrics.reset_metrics()
        val_metrics.reset_metrics()

        if config["save_model"]:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(
                unwrapped_model.state_dict(), f"{config['save_dir']}/model_{epoch}.pt"
            )

    accelerator.end_training()


if __name__ == "__main__":
    accelerator = Accelerator(log_with="wandb")

    args = parse_args()

    config = {}
    config.update({k: v for k, v in args.__dict__.items()})

    if config["wandb"] or config["wandb_entity"]:
        # if we set wandb_entity, we set to True automatically
        config["wandb"] = True
        config["wandb_entity"] = (
            config["wandb_entity"] if config["wandb_entity"] else getpass.getuser()
        )

    name = config["wandb_exp_name"] if "wandb_exp_name" in config else None
    accelerator.init_trackers(
        project_name=config["wandb_project"],
        config=config,
        init_kwargs={"wandb": {"entity": config["wandb_entity"], "name": name}},
    )

    if config["save_model"]:
        # create save dir with random name
        if not os.path.exists(config["save_dir"]):
            # only save once per server (not sure if needed?)
            if accelerator.is_local_main_process:
                os.makedirs(config["save_dir"])

    accelerator.print("| configs: ")
    for k, v in config.items():
        accelerator.print("  |", k, " : ", v)

    train(accelerator=accelerator, config=config)
