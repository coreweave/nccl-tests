#!/bin/bash
###
#SBATCH --job-name=nccl_test_allreduce
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=20:00
#SBATCH --output="%x_%j.out" # Use %x for job name and %j for slurm job ID in output file name
#SBATCH --exclusive # Use --exclusive to ensure no other jobs run on the same nodes


module load image-defaults

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0

# Relaxed ordering is fixed in NCCL 2.18.3+, but
# in NCCL 2.18.1 and earlier it should be disabled
# for H100s due to a bug. See:
# https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2181/release-notes/rel_2-18-1.html
# export NCCL_IB_PCI_RELAXED_ORDERING=0

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

srun /opt/nccl_tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 1
