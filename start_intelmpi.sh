#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name="openbioml"
#SBATCH --nodes=4
#SBATCH --ntasks-per-node 1 
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:8
#SBATCH --output=/fsx/home-zanussbaum/bio-chem-lm/logs/bio-chem-lm_latest_%A_%a.out  # Set this dir where you want slurm outs to go
#SBATCH --error=/fsx/home-zanussbaum/bio-chem-lm/logs/bio-chem-lm_latest_%A_%a.err  # Set this dir where you want slurm outs to go
#SBATCH --comment openbioml
#SBATCH --exclusive


module load intelmpi
source /opt/intel/mpi/latest/env/vars.sh
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export NCCL_PROTO=simple
export PATH=/opt/amazon/efa/bin:$PATH
export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

#export NCCL_ALGO=ring
export NCCL_DEBUG=DEBUG
#export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL

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


#export NCCL_P2P_DISABLE=1

#export NCCL_IBEXT_DISABLE=1
#export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"

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

mpirun -n $COUNT_NODE -perhost 1 /fsx/home-zanussbaum/bio-chem-lm/train.sh