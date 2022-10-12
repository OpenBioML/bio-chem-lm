import glob
import os
from copy import deepcopy

import yaml
from mup import make_base_shapes

from bio_lm.model.config import ElectraConfig
from bio_lm.model.discriminator import ElectraForPreTraining
from bio_lm.model.generator import ElectraForMaskedLM


def load_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def make_shapes(config, base_model, model_config, save_dir, generator=False):
    delta_config = deepcopy(config)
    delta_config.update(load_config(model_config))

    delta_config = ElectraConfig(**delta_config)
    if generator:
        delta_model = ElectraForMaskedLM(delta_config)
    else:
        delta_model = ElectraForPreTraining(delta_config)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    names = glob.glob(f"{save_dir}/*.bsh")
    num_shapes = len(names)
    filename = f"{save_dir}/shapes_{num_shapes}.bsh"

    make_base_shapes(base_model, delta_model, savefile=filename)

    return filename
