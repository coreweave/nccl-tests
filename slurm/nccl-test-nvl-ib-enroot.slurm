#!/bin/bash
###
# NCCL all_reduce_perf on a rack-scale NVL system over InfiniBand,
# running inside the published nccl-tests container via pyxis/enroot.
#
# Applies to NVL72-style nodes with 4 GPUs per node connected by an
# external NVLink switch and IB on the backend (e.g., gb200-4x).
# Use the `-roce` variant for the RoCE-backed NVL SKUs.
#
# `--segment=16` keeps all ranks within a single NVLink domain so the
# MNNVL transport stays inside one rack. Increase --nodes / --segment
# proportionally for multi-rack tests.
###
#SBATCH --job-name=nccl_test_allreduce
#SBATCH --nodes=16
#SBATCH --segment=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=20:00
#SBATCH --output="%x_%j.out"
#SBATCH --distribution=block:block # Align ranks with numa: rank0-1 = numa0, rank2-3 = numa1
#SBATCH --exclusive

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# Out Of Band is set to run over front-end ethernet.
# Backend is restricted to ibp* to avoid the frontend's RoCE interfaces.
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp

# NVL rack-scale settings: fail fast if NVLink is down (no shared-memory
# fallback) and enable CUMEM + MNNVL for the external NVSwitch path.
export NVIDIA_IMEX_CHANNELS=0
export NCCL_NET_GDR_C2C=1
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_SHM_DISABLE=0

# Disable UCX. NCCL does not use UCX; we restrict its transport to TCP
# and point it at the frontend so a stray UCX init can't crash the job.
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth0
export OMPI_MCA_coll_hcoll_enable=0
export PMIX_MCA_gds='^ds12'
export NCCL_NET_PLUGIN=none

# See https://github.com/coreweave/nccl-tests/pkgs/container/nccl-tests for available tags.
nccl_version="12.9.1-devel-ubuntu22.04-nccl2.30.4-1-111c25a"

# Container image cache on shared storage so we don't re-import per node.
CONTAINER_DIR="$(realpath -s images)"
mkdir -p "$CONTAINER_DIR"
fstype="$(stat '-fc%T' "$CONTAINER_DIR")"
if [ "$fstype" != "nfs" ] ; then
  echo 'You must specify a container directory that is mounted on all cluster nodes.' >&2
  exit 1
fi

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

echo "Using nodes: $SLURM_JOB_NODELIST"

# The images are built with Ubuntu 22.04+ and use PMIx, so non-root users
# need --no-container-remap-root and root users need --container-remap-root.
if [ "$(whoami)" = "root" ]; then
  cflag="--container-remap-root"
else
  cflag="--no-container-remap-root"
fi

srun --container-image="$CONTAINER_IMAGE" \
     --mpi=pmix "$cflag" --no-container-mount-home --kill-on-bad-exit=1 \
     /opt/nccl_tests/build/all_reduce_perf -b 16M -e 8G -f 2 -g 1
