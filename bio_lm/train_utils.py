import os
import numpy as np
import torch

import json
import yaml
from datetime import datetime
from mup import make_base_shapes

from bio_lm.model.electra.config import ElectraConfig
from bio_lm.model.electra.discriminator import ElectraForPreTraining
from bio_lm.model.electra.electra import Electra
from bio_lm.model.electra.generator import ElectraForMaskedLM

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from collections import OrderedDict

BASE = "model/configs/{model_type}/{size}"


ENDC = "\033[0m"
COLORS = ["\033[" + str(n) + "m" for n in list(range(91, 97)) + [90]]
RED = COLORS[0]
BLUE = COLORS[3]
CYAN = COLORS[5]
GREEN = COLORS[1]

name2color = {
    "disc_input": BLUE,
    "right": GREEN,
    "wrong": RED,
}


def load_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def make_shapes(base_size, delta_size, config, vocab_size, pad_id, mask_id, save_dir):
    disc_config = load_config(BASE.format(model_type="discriminator", size=delta_size))
    disc_config["mup"] = True
    disc_config["vocab_size"] = vocab_size

    for key in disc_config:
        if key in config:
            disc_config[key] = config[key]

    disc_model_config = ElectraConfig(**disc_config)
    discriminator = ElectraForPreTraining(disc_model_config)

    gen_config = load_config(BASE.format(model_type="generator", size=delta_size))
    gen_config["mup"] = True
    gen_config["vocab_size"] = vocab_size

    for key in gen_config:
        if key in config:
            gen_config[key] = config[key]

    gen_model_config = ElectraConfig(**gen_config)
    generator = ElectraForMaskedLM(gen_model_config)

    tie_weights(generator, discriminator)

    delta_electra = Electra(
        discriminator=discriminator,
        generator=generator,
        pad_token_id=pad_id,
        mask_token_id=mask_id,
        config=gen_model_config,
    )

    base_disc_config = load_config(
        BASE.format(model_type="discriminator", size=base_size)
    )

    for key in base_disc_config:
        if key in config:
            base_disc_config[key] = config[key]

    base_disc_config["mup"] = True
    base_disc_config["vocab_size"] = vocab_size

    base_disc_model_config = ElectraConfig(**base_disc_config)
    base_discriminator = ElectraForPreTraining(base_disc_model_config)

    base_gen_config = load_config(BASE.format(model_type="generator", size=base_size))
    for key in base_gen_config:
        if key in config:
            base_gen_config[key] = config[key]

    base_gen_config["mup"] = True
    base_gen_config["vocab_size"] = vocab_size

    base_gen_model_config = ElectraConfig(**base_gen_config)
    base_generator = ElectraForMaskedLM(base_gen_model_config)

    tie_weights(base_generator, base_discriminator)

    base_electra = Electra(
        discriminator=base_discriminator,
        generator=base_generator,
        pad_token_id=pad_id,
        mask_token_id=mask_id,
        config=gen_model_config,
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # make filename unique
    config_hash = hash(json.dumps(config, sort_keys=True))
    filename = f"{save_dir}/shapes_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}_{config_hash}.bsh"

    make_base_shapes(base_electra, delta_electra, savefile=filename)

    return filename


def tie_weights(generator, discriminator):
    generator.electra.embeddings.word_embeddings = (
        discriminator.electra.embeddings.word_embeddings
    )
    generator.electra.embeddings.position_embeddings = (
        discriminator.electra.embeddings.position_embeddings
    )
    generator.electra.embeddings.token_type_embeddings = (
        discriminator.electra.embeddings.token_type_embeddings
    )

    
def print_token_diff(tokens, tokenizer, labels, idx, name=None, prepend=""):
    color = name2color.get(name, None)

    indices = torch.nonzero(tokens != labels).to("cpu").tolist()

    decoded_tokens = tokenizer.batch_decode(tokens)

    first_pad = -1
    decoded = decoded_tokens[idx].split()

    for j in range(len(decoded)):
        if decoded[j] == "[PAD]":
            first_pad = j
            break

        if color:
            if [idx, j] in indices:
                decoded[j] = color + decoded[j] + ENDC

    print(prepend + ": " + " ".join(decoded[:first_pad]))


def print_pred_replaced(tokens, tokenizer, pred_replaced, labels, idx):
    right = name2color["right"]
    wrong = name2color["wrong"]

    decoded_tokens = tokenizer.batch_decode(tokens)

    first_pad = -1
    decoded = decoded_tokens[idx].split()

    for j in range(len(decoded)):
        if decoded[j] == "[PAD]":
            first_pad = j
            break

        if pred_replaced[idx, j]:
            if labels[idx, j]:
                decoded[j] = right + decoded[j] + ENDC
            else:
                decoded[j] = wrong + decoded[j] + ENDC

        elif labels[idx, j]:
            decoded[j] = wrong + decoded[j] + ENDC

    print("DISCRIMINATOR: " + ": " + " ".join(decoded[:first_pad]))

    
def standardize(inputs, mean, std):
    return {"target": (inputs["target"] - mean) / std}

def prune_state_dict(model_dir):
    """Remove problematic keys from state dictionary"""
    if not (model_dir and os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))):
        try:
            state_dict_path = hf_hub_download(model_dir, filename="pytorch_model.bin")
        except RepositoryNotFoundError:
            return None
    else:
        state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
    assert os.path.exists(
        state_dict_path
    ), f"No `pytorch_model.bin` file found in {model_dir}"
    loaded_state_dict = torch.load(state_dict_path)
    state_keys = loaded_state_dict.keys()
    keys_to_remove = [
        k for k in state_keys if k.startswith("regression") or k.startswith("norm")
    ]

    new_state_dict = OrderedDict({**loaded_state_dict})
    for k in keys_to_remove:
        del new_state_dict[k]
    return new_state_dict