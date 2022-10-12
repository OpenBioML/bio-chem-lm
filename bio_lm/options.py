import multiprocessing as mp
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    # env params
    parser.add_argument("--gpu", action="store_true")

    # data params
    parser.add_argument("--dataset_name", type=str, default="zpn/pubchem_selfies")
    parser.add_argument("--tokenizer_name", type=str, default="zpn/pubchem_selfies_tokenizer_wordlevel")
    parser.add_argument("--mask_prob", type=float, default=0.15)

    # model params
    parser.add_argument("--generator_config", type=str, default="configs/electra-generator-base.json")
    parser.add_argument("--discriminator_config", type=str, default="configs/electra-discriminator-base.json")
    parser.add_argument("--position_embedding_type", type=str, default="alibi")

    # mup params
    parser.add_argument("--disc_base_shapes", type=str, help="directory to save bsh for mup")
    parser.add_argument("--gen_base_shapes", type=str, help="directory to save file for mup")
    parser.add_argument("--mup", action="store_true")

    # optimizer params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)

    # training params
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_steps_per_epoch", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="exps/")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    parser.add_argument("--num_warmup_steps", type=int)
    parser.add_argument("--num_training_steps", type=int)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--global_clip_norm", type=float)
    parser.add_argument("--debug", action="store_true")

    # wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()
