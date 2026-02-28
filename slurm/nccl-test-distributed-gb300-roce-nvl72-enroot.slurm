#!/bin/bash
###
#SBATCH --job-name=nccl_test_allreduce
#SBATCH --nodes=16
#SBATCH --segment=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=20:00
#SBATCH --output="%x_%j.out" # Use %x for job name and %j for slurm job ID in output file name
#SBATCH --distribution=block:block # Better align ranks with numa affinities ( rank0-1 = numa0, rank2-3 = numa1 )
#SBATCH --exclusive # Use --exclusive to ensure no other jobs run on the same nodes

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# This variant uses RoCE networking with the Spectrum-X NCCL plugin.

# Spectrum-X NCCL plugin
export LD_LIBRARY_PATH=/opt/hpcx/nccl_spectrum-x_plugin/lib:$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=spcx

# RoCE / Spectrum-X tuning
export NCCL_IB_NET_LATENCY=50
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_NVLS_NCHANNELS=24
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_NVLSTREE_MAX_CHUNKSIZE=262144
export NCCL_NVLS_CHUNKSIZE=262144
export NCCL_IB_ADDR_FAMILY=AF_INET6
export NCCL_IB_ADDR_RANGE=fd02::/16
export NCCL_IB_TC=96

# Out Of Band is set to run over front-end ethernet.
# Backend is restricted to use ibp* interfaces.
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp

# Disable UCX
# Restrict the transport layer for UCX, it tries to use all transports by default, this forces it on TCP. NCCL does not use UCX at all.
# We explictly deactivate it to avoid initializing UCX by mistake as it can lead to crashes.
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth0
export OMPI_MCA_coll_hcoll_enable=0
export PMIX_MCA_gds='^ds12'

# Define nccl version we will test
# See https://github.com/coreweave/nccl-tests/pkgs/container/nccl-tests for available tags.

nccl_version="13.1.0-devel-ubuntu22.04-nccl2.29.2-1-0bf62c2"

# Create a directory to store container images. You can change this
# or use an existing directory below.

CONTAINER_DIR="$(realpath -s images)"
mkdir -p "$CONTAINER_DIR"
fstype="$(stat '-fc%T' "$CONTAINER_DIR")"
if [ "$fstype" != "nfs" ] ; then
  echo 'You must specify a container directory that is mounted on all cluster nodes.' >&2
  exit 1
fi

# Pull the container image, if not already pulled. For large parallel jobs, this
# will save time by not hitting the repository from each task. This will
# be executed once on the head node of the allocation.

CONTAINER_IMAGE="${CONTAINER_DIR}/nccl_${nccl_version}.sqsh"
if [ -f "$CONTAINER_IMAGE" ]; then
   echo "Container image for NCCL version $nccl_version already exists, no need to pull."
else
   echo "Pulling container image for NCCL version: $nccl_version and saving to $CONTAINER_IMAGE"
   if ! enroot import -o "$CONTAINER_IMAGE" "docker://ghcr.io#coreweave/nccl-tests:$nccl_version"; then
       echo "Failed to pull container image: ghcr.io/coreweave/nccl-tests:$nccl_version" >&2
       exit 1
   fi
fi

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

# When launching a pyxis job as a non-root user you need to use --no-container-remap-root and when launching as root
# you need to use --container-remap-root. This is because the container is built with Ubuntu 22.04+ and is built to use PMIx.

if [ "$(whoami)" = "root" ]; then
  cflag="--container-remap-root"
else
  cflag="--no-container-remap-root"
fi

# The srun command inherits the SBATCH flags set above
# and will be launched once for each task.

srun --container-image="$CONTAINER_IMAGE" \
     --mpi=pmix "$cflag" --no-container-mount-home --kill-on-bad-exit=1 \
     /opt/nccl_tests/build/all_reduce_perf -b 16M -e 8G -f 2 -g 1
