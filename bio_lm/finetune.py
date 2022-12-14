import getpass
import os

import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

from bio_lm.dataset import load_finetune_dataset
from bio_lm.metrics import MetricDict, format_metrics, name2metric, PROBLEM2METRICS
from bio_lm.model.classifier import ElectraForSequenceClassification
from bio_lm.options import parse_args_finetune

    
def finetune(accelerator, config):
    set_seed(config["seed"])
    accelerator.print(f"NUM GPUS: {accelerator.num_processes}")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    with accelerator.main_process_first():
        train_dataloader, problem_type, num_labels, class_weights, mean, std = load_finetune_dataset(config, tokenizer)
        val_dataloader, _, _, _, _, _ = load_finetune_dataset(config, tokenizer, split="validation", mean=mean, std=std)


    model_config = AutoConfig.from_pretrained(config["model_name"])
    model_config.num_labels = num_labels
    model_config.class_weights = class_weights
    model = ElectraForSequenceClassification.from_pretrained(config["model_name"], config=model_config) 

    device = accelerator.device
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config["lr"])

    if config["scheduler"]:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["num_warmup_steps"],
            num_training_steps=config["num_training_steps"],
        )

    metric_names = PROBLEM2METRICS[problem_type]

    train_metrics = MetricDict(
        ["loss"],
        name2metric={metric_name: name2metric[metric_name] for metric_name in metric_names},
        device=device,
    )
    val_metrics = MetricDict(
        ["loss"],
        name2metric={metric_name: name2metric[metric_name] for metric_name in metric_names},
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

    if config["patience"] is not None:
        patience = 0
        early_stopping_metric_name = config["metric_for_early_stopping"]
        maximize = True if early_stopping_metric_name in ["pearsonr", "auroc", "precision"] else False
        best_val = float("-inf") if maximize else float("inf")
        

    for epoch in range(config["num_epochs"]):
        for step, batch in enumerate(
            tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc="Training",
                disable=not accelerator.is_local_main_process,
            )
        ):
            if step == 5 and config["debug"]:
                break

            optimizer.zero_grad()
            model.train()

            outputs = model(**batch)
            logits = outputs["logits"]
            loss = outputs["loss"]
            accelerator.backward(loss)

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

            logits = accelerator.gather_for_metrics(logits)
            targets = accelerator.gather_for_metrics(batch["target"])

            loss = accelerator.gather_for_metrics(loss)
            metrics = {"loss": {"value": loss.detach().item()}}

            for metric in metric_names:
                if num_labels == 2:
                    metrics[metric] = {"preds": logits[:, 0], "target": targets}
                else:
                    metrics[metric] = {"preds": logits.squeeze(), "target": targets}

            train_metrics.update(metrics)

        with torch.no_grad():
            # we only evalute with N steps since there are 10M data points!!
            for i, batch in enumerate(
                tqdm(
                    val_dataloader,
                    desc="Validation",
                    total=len(val_dataloader),
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

                val_metrics.update(metrics)


        log_train = {
            f"train_{key}": value for key, value in train_metrics.compute().items()
        }
        log_val = {f"val_{key}": value for key, value in val_metrics.compute().items()}

        if config["wandb"]:
            accelerator.log({**log_train, **log_val})

        accelerator.print(format_metrics(log_train, "train", f" epoch {epoch} "))
        accelerator.print(format_metrics(log_val, "val", f" epoch {epoch} "))

        curr_val = log_val[f"val_{early_stopping_metric_name}"]
        if maximize:
            if curr_val > best_val:
                best_val = curr_val
                patience = 0
                accelerator.print(f"New best val: {best_val}")
            else:
                patience += 1
        else:
            if curr_val < best_val:
                best_val = curr_val
                patience = 0
                accelerator.print(f"New best val: {best_val}")
            else:
                patience += 1


        train_metrics.reset_metrics()
        val_metrics.reset_metrics()

        if config["save_model"]:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(
                unwrapped_model.state_dict(), f"{config['save_dir']}/model_{epoch}.pt"
            )

            
        if patience == config["patience"]:
            break

    accelerator.end_training()


if __name__ == "__main__":
    accelerator = Accelerator()#log_with="wandb")

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

    finetune(accelerator=accelerator, config=config)
