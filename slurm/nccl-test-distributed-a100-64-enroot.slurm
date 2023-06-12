#!/bin/bash
###
#SBATCH --job-name=nccl_test
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=a100:8
#SBATCH --time=10:00
#SBATCH --output="%x.out"
#SBATCH --output="%x.out" #-%j
#SBATCH --exclusive

module load image-defaults

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1

# log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

srun --container-image=ghcr.io/coreweave/nccl-tests:12.1.1-cudnn8-devel-ubuntu20.04-nccl2.18.1-1-29e3624 \
     --container-remap-root --no-container-mount-home \
     /opt/nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1 -w 1 -n 10
