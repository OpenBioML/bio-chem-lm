import multiprocessing as mp
from argparse import ArgumentParser


def parse_args():
    # fmt: off
    parser = ArgumentParser()

    # data params
    parser.add_argument("--dataset_name", type=str, default="zpn/pubchem_selfies")
    parser.add_argument("--tokenizer_name", type=str, default="zpn/pubchem_selfies_tokenizer_wordlevel_dissociation")
    parser.add_argument("--mask_prob", type=float, default=0.15)

    # model params
    parser.add_argument("--base_config_size", type=str, default="tiny.yaml", help="Default base size for when creating shapes file")
    parser.add_argument("--generator_config", type=str, default="model/configs/generator/tiny.yaml")
    parser.add_argument("--discriminator_config", type=str, default="model/configs/discriminator/tiny.yaml")
    parser.add_argument("--position_embedding_type", type=str, default="absolute")
    parser.add_argument("--output_mult", type=int, default=1)
    parser.add_argument("--prenorm", action="store_true")
    parser.add_argument("--embedding_norm_layer_type", type=str, default="layer_norm")
    parser.add_argument("--embedding_num_groups", type=int, default=1, help="Number of groups for embedding group norm")
    parser.add_argument("--attn_norm_layer_type", type=str, default="layer_norm")
    parser.add_argument("--attn_num_groups", type=int, default=1, help="Number of groups for attn group norm")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_dir", type=str, default="saved_models/")

    # mup params
    parser.add_argument("--base_shapes_dir", type=str, default="shapes", help="directory to save bsh for mup")
    parser.add_argument("--mup", action="store_true")
    parser.add_argument("--readout_zero_init", action="store_true")
    parser.add_argument("--query_zero_init", action="store_true")

    # optimizer params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--validation_batch_size", type=int, default=32)

    # training params
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_steps_per_epoch", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="exps/")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    parser.add_argument("--num_warmup_steps", type=int, default=None)
    parser.add_argument("--num_training_steps", type=int, default=None)
    parser.add_argument("--num_eval_steps", type=int, default=None)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--global_clip_norm", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--log_predictions", action="store_true")

    # wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    # fmt: on

    return parser.parse_args()


def parse_args_finetune():
    # fmt: off
    parser = ArgumentParser()

    # hf params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="zpn/pubchem_selfies_tokenizer_wordlevel_dissociation")
    parser.add_argument("--dataset_name", type=str, required=True)

    # training params
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="exps/")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    parser.add_argument("--num_warmup_steps", type=int, default=None)
    parser.add_argument("--num_training_steps", type=int, default=None)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--global_clip_norm", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--metric_for_early_stopping", type=str, default="loss")

    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_dir", type=str)

    # optimizer params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--validation_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size" , type=int, default=32)
    
    # wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    # fmt: on

    return parser.parse_args()
