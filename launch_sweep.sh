#!/bin/bash
#SBATCH --job-name="openfold"
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --output=/fsx/home-zanussbaum/bio-chem-lm/logs/sweep_latest_%A_%a.out  # Set this dir where you want slurm outs to go
#SBATCH --error=/fsx/home-zanussbaum/bio-chem-lm/logs/sweep_latest_%A_%a.err  # Set this dir where you want slurm outs to go
#SBATCH --comment openfold
#SBATCH --partition g40423

export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export WORLD_SIZE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

#wandb
export WANDB_DIR="/fsx/home-zanussbaum/bio-chem-lm/outputs/"
export WANDB_CACHE_DIR="/fsx/home-zanussbaum/.cache"
export WANDB_MODE="online"
export WANDB_START_METHOD="thread"

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${WORLD_SIZE}

module load cuda/11.7
source /fsx/home-zanussbaum/bio-chem-lm/env/bin/activate

cd /fsx/home-zanussbaum/bio-chem-lm/bio_lm

srun wandb agent zanussbaum/plm_electra/abztvsoh