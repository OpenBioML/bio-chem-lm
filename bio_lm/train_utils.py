import glob
import os

import yaml
from mup import make_base_shapes

from bio_lm.model.config import ElectraConfig
from bio_lm.model.discriminator import ElectraForPreTraining
from bio_lm.model.electra import Electra
from bio_lm.model.generator import ElectraForMaskedLM

BASE = "model/configs/{model_type}/{size}"


def load_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def make_shapes(base_size, delta_size, vocab_size, pad_id, mask_id, save_dir):
    disc_config = load_config(BASE.format(model_type="discriminator", size=delta_size))
    disc_config["mup"] = True
    disc_config["vocab_size"] = vocab_size
    disc_model_config = ElectraConfig(**disc_config)
    discriminator = ElectraForPreTraining(disc_model_config)

    gen_config = load_config(BASE.format(model_type="generator", size=delta_size))
    gen_config["mup"] = True
    gen_config["vocab_size"] = vocab_size
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
    base_disc_config["mup"] = True
    base_disc_config["vocab_size"] = vocab_size
    base_disc_model_config = ElectraConfig(**base_disc_config)
    base_discriminator = ElectraForPreTraining(base_disc_model_config)

    base_gen_config = load_config(BASE.format(model_type="generator", size=base_size))
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

    names = glob.glob(f"{save_dir}/*.bsh")
    num_shapes = len(names)
    filename = f"{save_dir}/shapes_{num_shapes}.bsh"

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
