#!/bin/bash
###
#SBATCH --job-name=nccl_test_allreduce
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=20:00
#SBATCH --output="%x_%j.out" # Use %x for job name and %j for slurm job ID in output file name
#SBATCH --distribution=block:block # Better align ranks with numa affinities ( rank0-3 = numa0, rank4-7 = numa1 )
#SBATCH --exclusive # Use --exclusive to ensure no other jobs run on the same nodes


. /usr/share/modules/init/bash
module load image-defaults

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

srun --mpi=pmix --kill-on-bad-exit=1 /opt/nccl_tests/build/all_reduce_perf -b 16M -e 8G -f 2 -g 1
