import getpass
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from bio_lm.dataset import load_finetune_dataset
from bio_lm.metrics import (PROBLEM2METRICS, MetricDict, format_metrics,
                            name2metric)
from bio_lm.model.classifier import ElectraForSequenceClassification
from bio_lm.options import parse_args_finetune


def evaluate(accelerator, config):
    set_seed(config["seed"])
    accelerator.print(f"NUM GPUS: {accelerator.num_processes}")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    with accelerator.main_process_first():
        # need to load train data to get mean and std
        _, problem_type, num_labels, class_weights, mean, std = load_finetune_dataset(
            config, tokenizer
        )
        test_dataloader, _, _, _, _, _ = load_finetune_dataset(
            config, tokenizer, split="test", mean=mean, std=std
        )

    model_config = AutoConfig.from_pretrained(config["model_name"])
    model_config.num_labels = num_labels
    model_config.class_weights = class_weights
    model = ElectraForSequenceClassification.from_pretrained(
        config["model_name"], config=model_config
    )

    device = accelerator.device
    model.to(device)

    metric_names = PROBLEM2METRICS[problem_type]

    test_metrics = MetricDict(
        ["loss"],
        name2metric={
            metric_name: name2metric[metric_name] for metric_name in metric_names
        },
        device=device,
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    with torch.no_grad():
        # we only evalute with N steps since there are 10M data points!!
        for i, batch in enumerate(
            tqdm(
                test_dataloader,
                desc="Test",
                total=len(test_dataloader),
                disable=not accelerator.is_local_main_process,
            )
        ):
            if i == 5 and config["debug"]:
                break

            model.eval()
            outputs = model(**batch)
            logits = outputs["logits"]
            loss = outputs["loss"]

            logits = accelerator.gather_for_metrics(logits)
            targets = accelerator.gather_for_metrics(batch["target"])

            loss = accelerator.gather_for_metrics(loss)
            metrics = {"loss": {"value": loss.detach().item()}}

            for metric in metric_names:
                if num_labels == 2:
                    metrics[metric] = {"preds": logits[:, 0], "target": targets}
                else:
                    metrics[metric] = {"preds": logits.squeeze(), "target": targets}

            test_metrics.update(metrics)

    log_test = {f"test_{key}": value for key, value in test_metrics.compute().items()}

    if config["wandb"]:
        accelerator.log({**log_test})

    accelerator.print(format_metrics(log_test, "test", f" epoch 0 "))
    accelerator.end_training()


if __name__ == "__main__":
    accelerator = Accelerator()  # log_with="wandb")

    args = parse_args_finetune()

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
                os.makedirs(config["save_dir"], exist_ok=True)

    accelerator.print("| configs: ")
    for k, v in config.items():
        accelerator.print("  |", k, " : ", v)

    evaluate(accelerator=accelerator, config=config)
