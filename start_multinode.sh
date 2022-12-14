#!/bin/bash
#SBATCH --job-name="openbioml"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node 1 
#SBATCH --output=/fsx/home-zanussbaum/bio-chem-lm/logs/bio-chem-lm_latest_%A_%a.out  # Set this dir where you want slurm outs to go
#SBATCH --error=/fsx/home-zanussbaum/bio-chem-lm/logs/bio-chem-lm_latest_%A_%a.err  # Set this dir where you want slurm outs to go
#SBATCH --partition g40n404
#SBATCH --comment openbioml
#SBATCH --requeue
#SBATCH --exclusive

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

export NCCL_DEBUG=INFO
export NCCL_PROTO=simple

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

export WANDB_DIR="/fsx/home-zanussbaum/bio-chem-lm/outputs/"
export WANDB_CACHE_DIR="/fsx/home-zanussbaum/.cache"
export WANDB_MODE="online"

export TORCH_SHOW_CPP_STACKTRACES=1

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export I_MPI_PORT_RANGE=12800:12804
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export I_MPI_HYDRA_BOOTSTRAP=ssh

echo go $COUNT_NODE
echo $HOSTNAMES
echo $MASTER_ADDR
echo $I_MPI_PORT_RANGE

srun -n $COUNT_NODE /fsx/home-zanussbaum/bio-chem-lm/train.sh