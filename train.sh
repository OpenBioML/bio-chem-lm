#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID

module load cuda/11.4
source /fsx/home-zanussbaum/bio-chem-lm/env/bin/activate

cd /fsx/home-zanussbaum/bio-chem-lm/bio_lm
`echo -e $HOSTNAMES  | python3 -c "import sys;lines = [f'{line.strip()} slots=8\n' for line in next(sys.stdin).split(' ')];open('hostfile','w').writelines(lines)"`

accelerate launch --use_deepspeed --num_processes $(( 8 * $COUNT_NODE )) --num_machines $COUNT_NODE --machine_rank $THEID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --zero_stage=2 --gradient_clipping=1 --gradient_accumulation_steps=1 --deepspeed_multinode_launcher standard --deepspeed_hostfile hostfile train.py --attn_norm_layer_type=group_norm --attn_num_groups=1 --base_config_size=tiny.yaml --discriminator_config=model/configs/discriminator/large.yaml --embedding_norm_layer_type=group_norm --embedding_num_groups=16 --generator_config=model/configs/generator/large.yaml --global_clip_norm=1 --lr=0.000746167727978618 --mup --num_epochs=20 --num_eval_steps=1000 --num_steps_per_epoch=5000 --output_mult=64 --position_embedding_type=rotary --prenorm --query_zero_init --readout_zero_init --train_batch_size=32 --validation_batch_size=64 --wandb --wandb_entity=zanussbaum --wandb_project=bio-chem-lm --save_model