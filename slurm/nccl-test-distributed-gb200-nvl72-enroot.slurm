#!/bin/bash
###
#SBATCH --job-name=nccl_test
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=20:00
#SBATCH --output="%x_%j.out"
#SBATCH --exclusive

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0

export NVIDIA_IMEX_CHANNELS=0
export NCCL_NVLS_ENABLE=0
export NCCL_NET_GDR_C2C=1
export PMIX_MCA_gds='^ds12'

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

# Save the container to a NFS mount for speed ups on large jobs
# --container-save=/mnt/home/user/nccl-tests.sqsh -> --container-image=/mnt/home/user/nccl-tests.sqsh

srun --container-image=ghcr.io#coreweave/nccl-tests:12.8.1-devel-ubuntu22.04-nccl2.26.2-1-0708d2e \
     --mpi=pmix --no-container-remap-root --container-mount-home \
     /opt/nccl_tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 1
