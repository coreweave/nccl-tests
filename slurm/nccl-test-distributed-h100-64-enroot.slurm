#!/bin/bash
###
#SBATCH --job-name=nccl_test
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=20:00
#SBATCH --output="%x.out"
#SBATCH --exclusive

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

srun --container-image=ghcr.io#coreweave/nccl-tests:12.4.1-cudnn-devel-ubuntu20.04-nccl2.21.5-1-85f9143 \
     --container-remap-root --no-container-mount-home \
     /opt/nccl_tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 1
