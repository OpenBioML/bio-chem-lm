from bio_lm.train_utils import load_config, tie_weights, make_shapes
from bio_lm.model.config import ElectraConfig
from bio_lm.model.discriminator import ElectraForPreTraining
from bio_lm.model.electra import Electra
from bio_lm.model.generator import ElectraForMaskedLM
from bio_lm.options import parse_args

from mup import set_base_shapes

import torch
from transformers import AutoTokenizer


if __name__ == "__main__":
    args = parse_args()

    config = {}
    config.update({k: v for k, v in args.__dict__.items()})

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    electra_shapes_filename = make_shapes(
                base_size=config["base_config_size"],
                delta_size="small.yaml",
                config=config,
                vocab_size=tokenizer.vocab_size,
                pad_id=tokenizer.pad_token_id,
                mask_id=tokenizer.mask_token_id,
                save_dir=config["base_shapes_dir"],
            )

    disc_config = load_config(config["discriminator_config"])
    # update keys in config with values from cli
    for key in disc_config:
        if key in config:
            disc_config[key] = config[key]

    disc_config["mup"] = True
    disc_config["vocab_size"] = tokenizer.vocab_size
    discriminator_config = ElectraConfig(**disc_config)
    discriminator = ElectraForPreTraining(discriminator_config)

    gen_config = load_config(config["generator_config"])
    # update keys in config with values from cli
    for key in gen_config:
        if key in config:
            gen_config[key] = config[key]

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

    old_params = {k: v.clone() for k, v in model.named_parameters()}

    missing = model.load_state_dict(
        torch.load('saved_models/model_5.pt', 
                map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))),

                
    assert missing

    new_params = {k: v.clone() for k, v in model.named_parameters()}


    for k in new_params:
        if torch.equal(old_params[k], new_params[k]):
            print(k)
            print(old_params[k])
            print(new_params[k])

            
    model.discriminator.save_pretrained("discriminator_5")

                