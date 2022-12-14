import getpass

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from bio_lm.dataset import load_finetune_dataset 
from bio_lm.metrics import (PROBLEM2METRICS, MetricDict, format_metrics,
                            name2metric)
from bio_lm.model.electra.classifier import ElectraForSequenceClassification
from bio_lm.model.roberta.classifier import RobertaForSequenceClassification, RobertaForRegression
from bio_lm.options import parse_args_evaluate


def evaluate(accelerator, config):
    set_seed(config["seed"])
    accelerator.print(f"NUM GPUS: {accelerator.num_processes}")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    with accelerator.main_process_first():
        (
            test_dataloader,
            problem_type,
            num_labels,
            class_weights,
        ) = load_finetune_dataset(config, tokenizer, split="test", use_selfies=True if config["model_type"] == "electra" else False)

    model_config = AutoConfig.from_pretrained(config["model_name"])
    model_config.num_labels = num_labels
    model_config.class_weights = class_weights
    if config["model_type"] == "electra":
        model = ElectraForSequenceClassification.from_pretrained(
            config["model_name"], config=model_config
        )
    elif config["model_type"] == "roberta" or config["model_type"] == "mtr":
        if problem_type == "regression":
            model_class = RobertaForRegression

        elif problem_type == "classification":
            model_class = RobertaForSequenceClassification
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

        model = model_class.from_pretrained(config["model_name"], config=model_config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

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
                    metrics[metric] = {"preds": F.softmax(logits, dim=1)[:, 1], "target": targets}
                else:
                    metrics[metric] = {"preds": logits.squeeze(), "target": targets}

            test_metrics.update(metrics)

    log_test = {f"test_{key}": value for key, value in test_metrics.compute().items()}

    if config["wandb"]:
        accelerator.log({**log_test})

    accelerator.print(format_metrics(log_test, "test", f" epoch 0 "))
    if config["wandb"]:
        accelerator.end_training()


if __name__ == "__main__":

    args = parse_args_evaluate()

    config = {}
    config.update({k: v for k, v in args.__dict__.items()})

    if config["wandb"] or config["wandb_entity"]:
        accelerator = Accelerator(log_with="wandb")
        # if we set wandb_entity, we set to True automatically
        config["wandb"] = True
        config["wandb_entity"] = (
            config["wandb_entity"] if config["wandb_entity"] else getpass.getuser()
        )
        if config["wandb_exp_name"] is None:
            config[
                "wandb_exp_name"
            ] = f"{config['model_name']}_{config['dataset_name']}"

        accelerator.init_trackers(
            project_name=config["wandb_project"],
            config=config,
            init_kwargs={
                "wandb": {
                    "entity": config["wandb_entity"],
                    "name": config["wandb_exp_name"],
                }
            },
        )
    else:
        accelerator = Accelerator()

    accelerator.print("| configs: ")
    for k, v in config.items():
        accelerator.print("  |", k, " : ", v)

    evaluate(accelerator=accelerator, config=config)
