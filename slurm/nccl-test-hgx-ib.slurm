#!/bin/bash
###
# NCCL all_reduce_perf on a standard 8-GPU HGX SKU over InfiniBand.
#
# Applies to any non-NVL HGX node with 8 GPUs + IB on the backend
# (e.g., gd-8xh100ib-i128, gd-8xh200ib-i128, gd-8xb200ib-i128).
# Use the dedicated NVL examples for rack-scale GB200/GB300.
#
# This variant runs the nccl-tests binaries pre-installed on the
# sunk bare-metal nodes via `module load image-defaults`; the
# `-enroot` variant runs the published container image instead.
###
#SBATCH --job-name=nccl_test_allreduce
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=20:00
#SBATCH --output="%x_%j.out"
#SBATCH --distribution=block:block # Align ranks with numa: rank0-3 = numa0, rank4-7 = numa1
#SBATCH --exclusive

. /usr/share/modules/init/bash
module load image-defaults

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# Out Of Band is set to run over front-end ethernet.
# Backend is restricted to ibp* to avoid the frontend's RoCE interfaces.
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp

# Disable UCX. NCCL does not use UCX; we restrict its transport to TCP
# and point it at the frontend so a stray UCX init can't crash the job.
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth0
export OMPI_MCA_coll_hcoll_enable=0
export PMIX_MCA_gds='^ds12'
export NCCL_NET_PLUGIN=none

echo "Using nodes: $SLURM_JOB_NODELIST"

srun --mpi=pmix --kill-on-bad-exit=1 /opt/nccl_tests/build/all_reduce_perf -b 16M -e 8G -f 2 -g 1
