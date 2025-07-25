#!/bin/bash
###
#SBATCH --job-name=nccl_test_allreduce
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=20:00
#SBATCH --output="%x_%j.out" # Use %x for job name and %j for slurm job ID in output file name
#SBATCH --exclusive # Use --exclusive to ensure no other jobs run on the same nodes

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1

# Define nccl version we will test
# See https://github.com/coreweave/nccl-tests/pkgs/container/nccl-tests for available tags.

nccl_version="12.9.1-devel-ubuntu22.04-nccl2.27.6-1-7c12c62"

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
# and will be launched once for each task (8 tasks on each of 8 nodes).

srun --container-image="$CONTAINER_IMAGE" \
     --mpi=pmix "$cflag" --no-container-mount-home --kill-on-bad-exit=1 \
     /opt/nccl_tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 1
