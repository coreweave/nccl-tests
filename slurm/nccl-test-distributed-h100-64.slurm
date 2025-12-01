#!/bin/bash
###
#SBATCH --job-name=nccl_test_allreduce
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=20:00
#SBATCH --output="%x_%j.out" # Use %x for job name and %j for slurm job ID in output file name
#SBATCH --distribution=block:block # Better align ranks with numa affinities ( rank0-3 = numa0, rank4-7 = numa1 )
#SBATCH --exclusive # Use --exclusive to ensure no other jobs run on the same nodes


. /usr/share/modules/init/bash
module load image-defaults

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# Out Of Band is set to run over front-end ethernet.
# Backend is restricted to use ibp* interfaces to ensure it doesn't try to use any RoCE interfaces from the frontend.

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp

# Disable UCX
# Restrict the transport layer for UCX, it tries to use all transports by default, this forces it on TCP. NCCL does not use UCX at all.
# We explictly deactivate it to avoid initializing UCX by mistake as it can lead to crashes.
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth0
export OMPI_MCA_coll_hcoll_enable=0
export PMIX_MCA_gds='^ds12'

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

srun --mpi=pmix --kill-on-bad-exit=1 /opt/nccl_tests/build/all_reduce_perf -b 16M -e 8G -f 2 -g 1
