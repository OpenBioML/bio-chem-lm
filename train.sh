#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script

echo myuser=`whoami`
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $SLURM_JOB_NODELIST
echo hostname = `hostname`
echo MASTER_ADDR= $SLURM_LAUNCH_NODE_IPADDR
echo MASTER_PORT= $MASTER_PORT
echo SLURM_JOB_NODELIST= $SLURM_JOB_NODELIST
echo SLURM_JOB_NUM_NODES= $SLURM_JOB_NUM_NODES
echo THEID=$SLURM_LOCALID

module load cuda/11.7
source /fsx/home-zanussbaum/bio-chem-lm/env/bin/activate

H=`hostname`
THEID=`nodelist_inflate --nodelist=$SLURM_JOB_NODELIST --hostname=$H`
echo $THEID

cd /fsx/home-zanussbaum/bio-chem-lm/bio_lm

nodelist_inflate --nodelist=$SLURM_JOB_NODELIST --write

accelerate launch --dynamo_backend=inductor --num_processes $(( 8 * $SLURM_JOB_NUM_NODES )) --num_machines $SLURM_JOB_NUM_NODES --machine_rank $THEID --main_process_ip $SLURM_LAUNCH_NODE_IPADDR --main_process_port $MASTER_PORT --deepspeed_multinode_launcher standard --deepspeed_hostfile hostnames --mixed_precision=bf16 --use_deepspeed --zero_stage=2 --offload_param_device=cpu --offload_optimizer_device=cpu --gradient_accumulation_steps=1 train_plm.py --lr=1e-4 --dataset_name=zpn/uniref50 --tokenizer_name=zpn/amino_acid_tokenizer --num_steps_per_epoch=1000  --global_clip_norm=1 --log_predictions --generator_config=model/configs/electra/generator/large.yaml --discriminator_config=model/configs/electra/discriminator/large.yaml --scheduler --num_warmup_steps=10000 --num_training_steps=100000 --num_epochs=100 --batch_size=8 --wandb --wandb_entity=zanussbaum --wandb_project=plm_electra --save_model --save_dir=saved_models/electra_large_plm_no_mup