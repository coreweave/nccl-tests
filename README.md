# NCCL for Distributed Training

CoreWeave supports the
[NVIDIA Collective Communication Library (NCCL)](https://developer.nvidia.com/nccl)
for powering multi-GPU and multi-node neural network training. NCCL underpins
the vast majority of all distributed training frameworks such as
[DeepSpeed](https://github.com/microsoft/DeepSpeed),
[PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
and [Horovod](https://horovod.readthedocs.io/en/stable/gpus_include.html).

NCCL is supported across all CoreWeave NVIDIA GPUs over Ethernet. In addition,
the specialized A100 HGX clusters are built to the design of NVIDIA DGX
SuperPODs, including
[NVIDIA Quantum InfiniBand](https://www.nvidia.com/en-us/networking/quantum2/)
networking and in-network collections using
[NVIDIA SHARP](https://docs.nvidia.com/networking/display/SHARPv270/Introduction)
to deliver the highest distributed training performance possible.

## Docker Images

This repository includes Dockerfiles that can be used directly or as a
template for your distributed training applications. The Dockerfiles include
the following components:

- NVIDIA [Mellanox OFED Driver](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/)
  userspace components. The kernel side is installed on our bare-metal nodes and
  does not need to be installed by users. The OFED drivers are necessary for
  optimized InfiniBand communication.
- NVIDIA [HPC-X](https://developer.nvidia.com/networking/hpc-x) which is a
  packaging of OpenMPI and UCX
- NVIDIA HPC-X OpenMPI compiled with external PMIx to
  enable [SLURM](https://slurm.schedmd.com/) integration
- NVIDIA [GDRCopy](https://developer.nvidia.com/gdrcopy) libraries leverage
  GPUDirect RDMA for improved GPU to host memory copy performance in certain
  applications. The kernel support for GDRCopy exists on CoreWeave's
  bare-metal nodes. GDRCopy is only supported on A100 training clusters.
- NVIDIA [NCCL SHARP Plugin](https://github.com/Mellanox/nccl-rdma-sharp-plugins)
  for SHARP support in NCCL
- NVIDIA [NCCL Tests](https://github.com/NVIDIA/nccl-tests) for verification
  and benchmarking purposes
- NVIDIA [DCGM](https://developer.nvidia.com/dcgm) for GPU tests and health
  checks
- NVIDIA [bandwidthTest](https://docs.nvidia.com/cuda/demo-suite/index.html#bandwidthTest)
  utility
- [RDMA Perftest](https://github.com/linux-rdma/perftest/) with GPUDirect
- OpenSSH server and related settings to enable images to easily be used as
  MPI Runners

CoreWeave
also [publishes images](https://hub.docker.com/r/coreweave/nccl-tests/tags)
built from these Dockerfiles that can be used as base for your own images.

| **Image Tag**                                                                     | **CUDA** | **NCCL** | **HPC-X** |
|-----------------------------------------------------------------------------------|----------|----------|-----------|
| ghcr.io/coreweave/nccl-tests:12.2.2-cudnn8-devel-ubuntu20.04-nccl2.18.5-1-a6a61ab | 12.2.2   | 2.18.5   | 2.16.0    |
| ghcr.io/coreweave/nccl-tests:12.1.1-cudnn8-devel-ubuntu20.04-nccl2.18.3-1-253a5b1 | 12.1.1   | 2.18.3   | 2.16.0    |
| ghcr.io/coreweave/nccl-tests:12.0.1-cudnn8-devel-ubuntu20.04-nccl2.18.5-1-a6a61ab | 12.0.1   | 2.18.5   | 2.16.0    |
| ghcr.io/coreweave/nccl-tests:11.8.0-cudnn8-devel-ubuntu20.04-nccl2.16.2-1-a6a61ab | 11.8.0   | 2.16.2   | 2.14.0    |
| ghcr.io/coreweave/nccl-tests:11.7.1-cudnn8-devel-ubuntu20.04-nccl2.14.3-1-a6a61ab | 11.7.1   | 2.14.3   | 2.14.0    |
| coreweave/nccl-tests:2022-09-28_16-34-19.392_EDT                                  | 11.6.2   | 2.12.0   | 2.12      |

## Running NCCL Tests

There are many sample jobs in this repo showing how to run distributed NCCL
tests, using the following workload managers:
 - [MPI Operator](https://github.com/kubeflow/mpi-operator)
 - [Slurm](https://slurm.schedmd.com/)

### MPI Operator

CoreWeave provides a managed instance of the
[MPI Operator](https://github.com/kubeflow/mpi-operator) to allow running
MPI Jobs in a container native fashion. No installation is required by the
user, simply execute an MPIJob manifest in your namespace.

Example manifests are provided in the `mpi-operator/` directory. There you'll
find the following examples of 64 GPU (8 node) runs:
 - [A40](./mpi-operator/nccl-test-distributed-a40-64-las1-mpijob.yaml)
 - [A100](./mpi-operator/nccl-test-distributed-a100-64-las1-mpijob.yaml)
 - [A100 with GDRCopy](./mpi-operator/nccl-test-distributed-a100-64-las1-gdrcopy-mpijob.yaml)
 - [A100 without Infiniband](./mpi-operator/nccl-test-distributed-a100-64-las1-no-ib-mpijob.yaml)
 - [A100 with SHARP](./mpi-operator/nccl-test-distributed-a100-64-las1-sharp-mpijob.yaml)
 - [H100](./mpi-operator/nccl-test-distributed-h100-64-las1-mpijob.yaml)
 - [H100 with SHARP](./mpi-operator/nccl-test-distributed-h100-64-las1-sharp-mpijob.yaml)

#### Running Jobs

To start the NCCL test, apply the sample manifest into your namespace with
`kubectl`:


```bash
$ kubectl apply -f nccl-test-distributed-h100-64-las1-sharp-mpijob.yaml
$ kubectl get pods
nccl-test-64-launcher-lnnrw   1/1     Running   0          14s
nccl-test-64-worker-0         1/1     Running   0          16s
nccl-test-64-worker-1         1/1     Running   0          16s
nccl-test-64-worker-10        1/1     Running   0          15s
...
$ kubectl logs -f -l=training.kubeflow.org/job-role=launcher
# nThread 1 nGpus 1 minBytes 4 maxBytes 2147483648 step: 2(factor) warmup iters: 50 iters: 50 validation: 1 
#
...
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
   536870912     134217728     float     sum      -1   2984.6  179.88  356.01      0   2979.7  180.18  356.60      0
  1073741824     268435456     float     sum      -1   5808.0  184.87  365.90      0   5882.2  182.54  361.28      0
  2147483648     536870912     float     sum      -1    11163  192.37  380.73      0    11203  191.70  379.40      0
  4294967296    1073741824     float     sum      -1    22181  193.63  383.23      0    22570  190.29  376.62      0
  8589934592    2147483648     float     sum      -1    43980  195.31  386.56      0    44094  194.81  385.56      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 373.187 
#
```

Before running a new instance of a test, delete the old with
`kubectl delete mpijob <job name>` or `kubectl delete mpijob --all`. Please
note that it is important to wait for all pods from an earlier job to finish
terminating before starting a new job with the same name.

### Slurm

CoreWeave provides a way to deploy a slurm cluster on top of our managed
kubernetes cluster using a tool called `sunk`.

Example `SBATCH` scripts are provided in the `slurm/` directory. There you'll
find the following examples of 64 GPU (8 node) runs:
 - [A100 without enroot](./slurm/nccl-test-distributed-a100-64.slurm)
 - [A100 with enroot](./slurm/nccl-test-distributed-a100-64-enroot.slurm)
 - [H100 without enroot](./slurm/nccl-test-distributed-h100-64.slurm)
 - [H100 with enroot](./slurm/nccl-test-distributed-h100-64-enroot.slurm)
 - [H100 with enroot and SHARP](./slurm/nccl-test-distributed-h100-64-enroot-sharp.slurm)

#### Running Jobs

To submit the jobs on a slurm cluster, first copy the scripts onto the login
node.

Various parameters are set by the scripts, but make sure to specify the
desired partition when submitting the job.

To start the NCCL test, submit the job via `sbatch`:

```bash
export PARTITION=<enter partition>
sbatch --partition="$PARTITION" nccl-test-distributed-a100-64.slurm
```

The logs will be written to `./nccl_test.out`.

**Note:** The jobs that don't use enroot rely on `nccl-tests` being installed
at `/opt/nccl-tests`, which will be true of every `sunk` cluster.

#### Enroot

[Enroot](https://github.com/nvidia/enroot) is a tool that enables running
unprivileged containers. In combination with
[pyxis](https://github.com/NVIDIA/pyxis), a slurm container plugin, you can
run slurm jobs inside of docker images.

There are additional parameters enabled by
[pyxis](https://github.com/NVIDIA/pyxis), but in these example scripts it gets
used via `srun`'s `--container-image` parameter. This prevents having to
install the script and its requirements on all compute nodes.

**Note:** You can specify the container image in an `sbatch`, but all the
commands will be then run from inside the container. Therefore, we recommend
only specifying the container image in any subsequent `srun` calls.

## Running DeepSpeed Training Jobs

Both of the workload managers can be used to run DeepSpeed based distributed
training jobs similarly to how the NCCL test jobs are run. They both will
create the MPI hostsfile for you, and DeepSpeed can simply be run as a command
like you would with a manual hostsfile setup.

## GDRCopy

[GDRCopy](https://developer.nvidia.com/gdrcopy) can be enabled to improve CPU
to GPU memory communication in certain use cases. GDRCopy is supported in NCCL
using a hidden environment variable `NCCL_GDRCOPY_ENABLE`. In our testing,
performance improvements for regular NCCL allreduce workloads have not been
measured. We do not recommend enabling GDRCopy for NCCL without performing
adequate benchmarks to ensure that performance is improved. It is noted in the
GDRCopy documentation that performance in some cases is degraded instead of
improved.
