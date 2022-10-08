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
    parser.add_argument("--vocab_size", type=int, default=30522)

    # model params
    parser.add_argument("--generator_config", type=str, default="configs/electra-generator-base.json")
    parser.add_argument("--discriminator_config", type=str, default="configs/electra-discriminator-base.json")
    parser.add_argument("--position_embedding_type", type=str, default="alibi")
    
    # optimizer params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)

    # training params
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_steps_per_epoch", type=int, default=1000)
    parser.add_argument("--mup", action="store_true")
    parser.add_argument("--output_dir", type=str, default="exps/")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    parser.add_argument("--n_warmup_steps", type=int, default=1000)
    parser.add_argument("--global_clip_norm", type=float, default=1.0)

    # wandb args
    parser.add_argument("--wandb_project", type=str, default="openbio-ml")
    parser.add_argument("--wandb_entity", type=str, default="zanussbaum")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    
    return parser.parse_args()