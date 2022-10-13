import getpass
from functools import partial

from accelerate import Accelerator
import torch
from datasets import load_dataset
from mup import MuAdam, set_base_shapes
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
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
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    train_dataloader = load_data(config, tokenizer)

    val_dataloader = load_data(config, tokenizer, split="validation")

    if config["mup"]:
        disc_config = load_config(config["discriminator_config"])
        disc_config["mup"] = True
        disc_config["vocab_size"] = tokenizer.vocab_size
        discriminator_config = ElectraConfig(**disc_config)
        discriminator = ElectraForPreTraining(discriminator_config)

        disc_filename = make_shapes(
            disc_config,
            discriminator,
            model_config="model/configs/discriminator/small.yaml",
            save_dir=config["disc_base_shapes"],
            generator=False,
        )
        set_base_shapes(discriminator, disc_filename)

        gen_config = load_config(config["generator_config"])
        gen_config["mup"] = True
        gen_config["vocab_size"] = tokenizer.vocab_size
        generator_config = ElectraConfig(**gen_config)
        generator = ElectraForMaskedLM(generator_config)

        gen_filename = make_shapes(
            gen_config,
            generator,
            model_config="model/configs/generator/small.yaml",
            save_dir=config["gen_base_shapes"],
            generator=True,
        )

        set_base_shapes(generator, gen_filename)
    else:
        disc_config = load_config(config["discriminator_config"])
        discriminator_config = ElectraConfig(disc_config)
        discriminator = ElectraForPreTraining(discriminator_config)

        discriminator.apply(
                partial(
                    model._init_weights,
                    readout_zero_init=discriminator_config,
                    query_zero_init=config["query_zero_init"],
                )
            )

        gen_config = load_config(config["generator_config"])
        generator_config = ElectraConfig(**gen_config)
        generator = ElectraForMaskedLM(generator_config)

        generator.apply(
                partial(
                    model._init_weights,
                    readout_zero_init=generator_config.readout_zero_init,
                    query_zero_init=config["query_zero_init"],
                )
            )

    accelerator = Accelerator()
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
        ]
    )
    val_metrics = MetricDict(
        [
            "loss",
            "mlm_loss",
            "disc_loss",
            "gen_acc",
            "disc_acc",
        ]
    )

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    train_dataloader = iter(train_dataloader)

    for epoch in range(config["num_epochs"]):
        for step in tqdm(range(config["num_steps_per_epoch"])):
            if step == 5 and config["debug"]:
                break

            optimizer.zero_grad()
            batch = next(train_dataloader)
            model.train()

            loss = model(**batch)
            accelerator.backward(loss["loss"])

            if config["global_clip_norm"]:
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
            for i, batch in enumerate(val_dataloader):
                if i == 5 and config["debug"]:
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
            wandb.log({**log_train, **log_val})

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
        entity = (
            config["wandb_entity"]
            if "wandb_entity" in config
            else f"{getpass.getuser()}"
        )
        name = config["wandb_exp_name"] if "wandb_exp_name" in config else None
        wandb.init(project="bio-chem-lm", entity=entity, name=name)
        wandb.config.update(config)

    print("| configs: ")
    for k, v in config.items():
        print("  |", k, " : ", v)

    train(config=config)
