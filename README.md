# NCCL for Distributed Training

CoreWeave supports the
[NVIDIA Collective Communication Library (NCCL)](https://developer.nvidia.com/nccl)
for powering multi-GPU and multi-node neural network training. NCCL underpins
the vast majority of all distributed training frameworks such as
[DeepSpeed](https://github.com/microsoft/DeepSpeed),
[PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
and [Horovod](https://horovod.readthedocs.io/en/stable/gpus_include.html).

NCCL is supported across CoreWeave NVIDIA GPUs over Ethernet and InfiniBand. In addition,
the specialized GB200 NVL72 clusters are built with
[NVIDIA Quantum-X800 InfiniBand](https://www.nvidia.com/en-us/networking/products/infiniband/quantum-x800/)
networking and in-network collections using
[NVIDIA SHARP](https://docs.nvidia.com/networking/display/sharpv300/introduction)
to deliver the highest distributed training performance possible.

* [NCCL for Distributed Training](#nccl-for-distributed-training)
   * [Docker Images](#docker-images)
   * [Running NCCL Tests](#running-nccl-tests)
      * [MPI Operator](#mpi-operator)
         * [Running Jobs](#running-jobs)
      * [Slurm](#slurm)
         * [Running Jobs](#running-jobs-1)
         * [Enroot](#enroot)
   * [Running DeepSpeed Training Jobs](#running-deepspeed-training-jobs)
   * [GDRCopy](#gdrcopy)
   * [Expected Performance](#expected-performance)
      * [GB200](#gb200)
         * [Single Rack](#single-rack)
         * [2 Racks](#2-racks)
         * [20 Racks](#20-racks)

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
   bare-metal nodes.
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
also [publishes images](https://github.com/coreweave/nccl-tests/pkgs/container/nccl-tests)
built from these Dockerfiles that can be used as base for your own images.  
The images below include **NCCL v2.27.6-1**, **HPC-X v2.23**, and **cuDNN v9.10.2.21-1**.  
Each image is multi-arch, and can be used for both `linux/amd64` and `linux/arm64` containers.
Compute capabilities up to Blackwell (10.0) are supported.

| **Image Tag**                                                              | **Ubuntu** | **CUDA** |
|----------------------------------------------------------------------------|------------|----------|
| ghcr.io/coreweave/nccl-tests:12.9.1-devel-ubuntu22.04-nccl2.27.6-1-7c12c62 | 22.04      | 12.9.1   |
| ghcr.io/coreweave/nccl-tests:12.8.1-devel-ubuntu22.04-nccl2.27.6-1-7c12c62 | 22.04      | 12.8.1   |
| ghcr.io/coreweave/nccl-tests:12.6.3-devel-ubuntu22.04-nccl2.27.6-1-7c12c62 | 22.04      | 12.6.3   |
| ghcr.io/coreweave/nccl-tests:12.4.1-devel-ubuntu22.04-nccl2.27.6-1-7c12c62 | 22.04      | 12.4.1   |
| ghcr.io/coreweave/nccl-tests:12.2.2-devel-ubuntu22.04-nccl2.27.6-1-7c12c62 | 22.04      | 12.2.2   |
| ghcr.io/coreweave/nccl-tests:12.9.1-devel-ubuntu20.04-nccl2.27.6-1-7c12c62 | 20.04      | 12.9.1   |
| ghcr.io/coreweave/nccl-tests:12.8.1-devel-ubuntu20.04-nccl2.27.6-1-7c12c62 | 20.04      | 12.8.1   |
| ghcr.io/coreweave/nccl-tests:12.6.3-devel-ubuntu20.04-nccl2.27.6-1-7c12c62 | 20.04      | 12.6.3   |
| ghcr.io/coreweave/nccl-tests:12.4.1-devel-ubuntu20.04-nccl2.27.6-1-7c12c62 | 20.04      | 12.4.1   |
| ghcr.io/coreweave/nccl-tests:12.2.2-devel-ubuntu20.04-nccl2.27.6-1-7c12c62 | 20.04      | 12.2.2   |

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
- [GB200 NVL72](./mpi-operator/nccl-test-distributed-gb200-nvl72-mpijob.yaml)

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
Kubernetes cluster using a tool called `sunk`.

Example `SBATCH` scripts are provided in the `slurm/` directory. There you'll
find the following examples of 64 GPU (8 node) runs:

- [A100 without enroot](./slurm/nccl-test-distributed-a100-64.slurm)
- [A100 with enroot](./slurm/nccl-test-distributed-a100-64-enroot.slurm)
- [H100 without enroot](./slurm/nccl-test-distributed-h100-64.slurm)
- [H100 with enroot](./slurm/nccl-test-distributed-h100-64-enroot.slurm)
- [H100 with enroot and SHARP](./slurm/nccl-test-distributed-h100-64-enroot-sharp.slurm)
- [GB200 with enroot](./slurm/nccl-test-distributed-gb200-nvl72-enroot.slurm)

#### Running Jobs

To submit the jobs on a slurm cluster, first copy the scripts onto the login
node.

Various parameters are set by the scripts, but make sure to specify the
desired partition when submitting the job.

To start the NCCL test, submit the job via `sbatch`:

```bash
export PARTITION=<enter partition>
sbatch --partition="$PARTITION" nccl-test-distributed-h100-64.slurm
```

You can also easily override the number of nodes the test will use. The following will use 4 nodes
instead of 8:

```bash
sbatch --partition="$PARTITION" -N 4 nccl-test-distributed-h100-64.slurm
```

The logs will be written to `./nccl_test_jobID.out`.

**Note:** The jobs that don't use enroot rely on `nccl-tests` being installed
at `/opt/nccl-tests`, which will be true on the compute nodes of every `sunk` cluster. The login node will probably *not* have this directory.

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
commands will then run from inside the container. Therefore, we recommend
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

## Expected Performance

The following results show the performance of the example jobs run on CoreWeave clusters.

> [!WARNING]
> Performance can vary across NCCL versions, changes to environment variables and even run to run.
> Keep this in mind when comparing runs.

### GB200

The following runs used NCCL `2.26.2`.

#### Single Rack

```bash
#  Rank 71 Group  0 Pid  14840 on slurm-gb200-207-171 device  3 [0x01] NVIDIA Graphics Device
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
   536870912     134217728     float     sum      -1   1806.7  297.16  586.06      0   1835.9  292.43  576.74      0
  1073741824     268435456     float     sum      -1   3108.8  345.38  681.17      0   2924.7  367.13  724.06      0
  2147483648     536870912     float     sum      -1   5589.1  384.22  757.78      0   5474.6  392.26  773.62      0
  4294967296    1073741824     float     sum      -1    10094  425.49  839.16      0    10141  423.54  835.32      0
  8589934592    2147483648     float     sum      -1    20299  423.16  834.57      0    20006  429.37  846.82      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 745.53 
```

#### 2 Racks

```bash
#  Rank 143 Group  0 Pid  14840 on slurm-gb200-207-171 device  3 [0x01] NVIDIA Graphics Device
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
   536870912     134217728     float     sum      -1   2156.8  248.92  493.94      0   2093.7  256.42  508.83      0
  1073741824     268435456     float     sum      -1   3574.4  300.40  596.10      0   3565.5  301.15  597.59      0
  2147483648     536870912     float     sum      -1   6264.0  342.83  680.30      0   6258.4  343.14  680.91      0
  4294967296    1073741824     float     sum      -1    11469  374.47  743.09      0    11435  375.61  745.36      0
  8589934592    2147483648     float     sum      -1    21493  399.65  793.06      0    21525  399.06  791.88      0
 17179869184    4294967296     float     sum      -1    42067  408.40  810.41      0    41557  413.41  820.36      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 688.487 
```

#### 20 Racks

```bash
#  Rank 1439 Group  0 Pid  21082 on slurm-gb200-218-073 device  3 [0x01] NVIDIA GB200
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)    
   536870912     134217728     float     sum      -1   7108.4   75.53  150.95      0   7145.8   75.13  150.16      0
  1073741824     268435456     float     sum      -1   9805.7  109.50  218.85      0   9819.6  109.35  218.54      0
  2147483648     536870912     float     sum      -1    14980  143.36  286.52      0    15087  142.34  284.49      0
  4294967296    1073741824     float     sum      -1    24782  173.31  346.38      0    24975  171.97  343.71      0
  8589934592    2147483648     float     sum      -1    45004  190.87  381.48      0    44930  191.19  382.11      0
 17179869184    4294967296     float     sum      -1    84625  203.01  405.74      0    84828  202.53  404.77      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 297.808 
```