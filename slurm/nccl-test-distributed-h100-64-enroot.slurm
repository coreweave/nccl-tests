#!/bin/bash
### Submit this script from a VAST mounted shared directory.
###
#SBATCH --job-name=nccl_test
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=20:00
#SBATCH --output="%x.out"
#SBATCH --exclusive
#SBATCH --kill-on-bad-exit=1

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0

# Define tags for nccl version we will test 
# See https://github.com/coreweave/nccl-tests/pkgs/container/nccl-tests for available tags.

nccl_version="12.9.1-devel-ubuntu22.04-nccl2.27.5-1-d5a135d"

# Create a directory to store container images
fstype=`df -T .| awk '{print $2}' | tail -1`
if [ $fstype != "nfs" ] ; then
  echo "This script must be run in a directory that is mounted on all cluster nodes."
  exit 1
fi
mkdir -p images

# Pull the container image, if not already pulled. For large parallel jobs, this
# will save time by not hitting the repository from each task.
if [ -f images/nccl_${tag}.sqsh ]; then
   echo "Container image for NCCL version $tag already exists, no need to pull."
    continue
fi

CONTAINER_IMAGE=`pwd`/images/nccl_${nccl_version}.sqsh
echo "Pulling container image for NCCL version: $nccl_version and saving to $CONTAINER_IMAGE"
enroot import docker://ghcr.io/coreweave/nccl-tests:$nccl_version \
  -o $CONTAINER_IMAGE

done

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

# When launching a pyxis job as a non-root user you need to use --no-container-remap-root and when launching as root
# you need to use --container-remap-root. This is becuse the container is built with Ubuntu 22.04+ and is built to use PMIx. 

if [ $(whoami) == "root" ]; then
  cflag="--container-remap-root"
else 
  cflag="--no-container-remap-root"
fi


srun --container-image=$CONTAINER_IMAGE \
     $cflag --no-container-mount-home \
     /opt/nccl_tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 1
