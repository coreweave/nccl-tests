# NCCL for Distributed Training

CoreWeave supports the [NVIDIA Collective Communication Library (NCCL)](https://developer.nvidia.com/nccl) for powering multi-GPU and multi-node neural network training. NCCL underpins the vast majority of all distributed training frameworks such as [DeepSpeed](https://github.com/microsoft/DeepSpeed), [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html) and [Horovod](https://horovod.readthedocs.io/en/stable/gpus_include.html).

NCCL is supported across all CoreWeave NVIDIA GPUs over Ethernet. In addition, the specialized A100 HGX clusters are built to the design of NVIDIA DGX SuperPODs, including [NVIDIA Quantum InfiniBand](https://www.nvidia.com/en-us/networking/quantum2/) networking and in-network collections using [NVIDIA SHARP](https://docs.nvidia.com/networking/display/SHARPv270/Introduction) to deliver the highest distributed training performance possible.

## Docker Images
This repository includes Dockerfiles that can be used directly or as a template for your distributed training applications. The Dockerfiles include the following components:
- NVIDIA [Mellanox OFED Driver](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/) userspace components. The kernel side is installed on our bare-metal nodes and does not need to be installed by users. The OFED drivers are necessary for optimized InfiniBand communication.
- NVIDIA [HPC-X](https://developer.nvidia.com/networking/hpc-x) which is a packaging of OpenMPI and UCX
- NVIDIA HPC-X OpenMPI compiled with external PMIx to enable [SLURM](https://slurm.schedmd.com/) integration
- NVIDIA [GDRCopy](https://developer.nvidia.com/gdrcopy) libraries leverage GPUDirect RDMA for improved GPU to host memory copy performance in certain applications. The kernel support for GDRCopy exists on CoreWeave's bare-metal nodes. GDRCopy is only supported on A100 training clusters.
- NVIDIA [NCCL SHARP Plugin](https://github.com/Mellanox/nccl-rdma-sharp-plugins) for SHARP support in NCCL
- NVIDIA [NCCL Tests](https://github.com/NVIDIA/nccl-tests) for verification and benchmarking purposes
- NVIDIA [DCGM](https://developer.nvidia.com/dcgm) for GPU tests and health checks
- NVIDIA [bandwidthTest](https://docs.nvidia.com/cuda/demo-suite/index.html#bandwidthTest) utility
- [RDMA Perftest](https://github.com/linux-rdma/perftest/) with GPUDirect
- OpenSSH server and related settings to enable images to easily be used as MPI Runners

CoreWeave also [publishes images](https://hub.docker.com/r/coreweave/nccl-tests/tags) built from these Dockerfiles that can be used as base for your own images.

| **Image Tag** | **CUDA** | **NCCL** | **HPC-X** |
|---------------|----------|----------|-----------|
| ghcr.io/coreweave/nccl-tests:12.0.0-devel-ubuntu20.04-nccl2.16.2-1-45d6ec9 | 12.0.0   | 2.16.2   | 2.13.1    |
| ghcr.io/coreweave/nccl-tests:11.8.0-devel-ubuntu20.04-nccl2.16.2-1-45d6ec9 | 11.8.0   | 2.16.2   | 2.13.1    |
| ghcr.io/coreweave/nccl-tests:11.7.1-devel-ubuntu20.04-nccl2.14.3-1-45d6ec9 | 11.7.1   | 2.14.3   | 2.13.1    |
| coreweave/nccl-tests:2022-09-28_16-34-19.392_EDT            | 11.6.2   | 2.12.0   | 2.12      |

## Running NCCL Tests
CoreWeave provides a managed instance of the [MPI Operator](https://github.com/kubeflow/mpi-operator) to allow running MPI Jobs in a container native fashion. No installation is required by the user, simply execute an MPIJob manifest in your namespace.

```
$ kubectl apply -f nccl-test-distributed-128-las1-mpijob.yaml
$ kubectl get pods
nccl-test-128-launcher-lnnrw   1/1     Running   0          14s
nccl-test-128-worker-0         1/1     Running   0          16s
nccl-test-128-worker-1         1/1     Running   0          16s
nccl-test-128-worker-10        1/1     Running   0          15s
...
$ kubectl logs -f -l=training.kubeflow.org/job-role=launcher
# nThread 1 nGpus 1 minBytes 4 maxBytes 2147483648 step: 2(factor) warmup iters: 50 iters: 50 validation: 1 
#
# Using devices
#   Rank  0 Pid     33 on nccl-test-128-worker-0 device  0 [0x27] NVIDIA A100-SXM4-80GB
#   Rank  1 Pid     34 on nccl-test-128-worker-0 device  1 [0x2a] NVIDIA A100-SXM4-80GB
#   Rank  2 Pid     35 on nccl-test-128-worker-0 device  2 [0x51] NVIDIA A100-SXM4-80GB
#   Rank  3 Pid     36 on nccl-test-128-worker-0 device  3 [0x57] NVIDIA A100-SXM4-80GB
#   Rank  4 Pid     37 on nccl-test-128-worker-0 device  4 [0x9e] NVIDIA A100-SXM4-80GB
...
g2f4e7c:0:588 - comm.c:392] INFO [group#:0] group id:0 tree idx:0 tree_type:LLT rail_idx:0 group size:4 quota: (osts:8 user_data_per_ost:1024) mgid: (subnet prefix:0xff12a01bfe800000 interface id:0x940000000000) mlid:c007
[g2f4e7c:0:588 - comm.c:392] INFO [group#:1] group id:0 tree idx:1 tree_type:SAT rail_idx:0 group size:4 quota: (osts:64 user_data_per_ost:0) mgid: (subnet prefix:0x0 interface id:0x0) mlid:0
#
#                                                       out-of-place                       in-place          
#       size         count      type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum    58.23    0.00    0.00  0e+00    56.02    0.00    0.00  0e+00
          16             4     float     sum    56.27    0.00    0.00  2e-07    55.45    0.00    0.00  2e-07
          32             8     float     sum    55.56    0.00    0.00  4e-07    55.57    0.00    0.00  4e-07
          64            16     float     sum    55.06    0.00    0.00  4e-07    55.01    0.00    0.00  4e-07
         128            32     float     sum    56.05    0.00    0.00  4e-07    55.68    0.00    0.00  4e-07
         256            64     float     sum    56.15    0.00    0.01  4e-07    56.15    0.00    0.01  2e-07
         512           128     float     sum    54.42    0.01    0.02  2e-07    55.37    0.01    0.02  1e-07
        1024           256     float     sum    56.74    0.02    0.03  5e-07    56.42    0.02    0.04  5e-07
        2048           512     float     sum    60.48    0.03    0.07  5e-07    61.71    0.03    0.06  5e-07
        4096          1024     float     sum    63.54    0.06    0.12  5e-07    62.65    0.07    0.13  5e-07
        8192          2048     float     sum    65.44    0.13    0.24  5e-07    65.00    0.13    0.24  5e-07
       16384          4096     float     sum    71.04    0.23    0.45  5e-07    70.11    0.23    0.45  5e-07
       32768          8192     float     sum    79.15    0.41    0.80  5e-07    77.87    0.42    0.82  5e-07
       65536         16384     float     sum    90.06    0.73    1.41  5e-07    89.23    0.73    1.42  5e-07
      131072         32768     float     sum    104.8    1.25    2.42  5e-07    98.90    1.33    2.57  5e-07
      262144         65536     float     sum    110.2    2.38    4.61  5e-07    106.7    2.46    4.76  5e-07
      524288        131072     float     sum    132.1    3.97    7.69  5e-07    133.3    3.93    7.62  5e-07
     1048576        262144     float     sum    143.4    7.31   14.17  5e-07    143.1    7.33   14.20  5e-07
     2097152        524288     float     sum    197.7   10.61   20.56  5e-07    193.2   10.85   21.03  5e-07
     4194304       1048576     float     sum    220.1   19.05   36.92  5e-07    220.2   19.05   36.91  5e-07
     8388608       2097152     float     sum    297.5   28.20   54.63  5e-07    297.5   28.20   54.63  5e-07
    16777216       4194304     float     sum    447.8   37.47   72.59  5e-07    449.8   37.30   72.26  5e-07
    33554432       8388608     float     sum    787.8   42.59   82.52  5e-07    778.5   43.10   83.51  5e-07
    67108864      16777216     float     sum   1399.2   47.96   92.93  5e-07   1396.9   48.04   93.08  5e-07
   134217728      33554432     float     sum   2619.6   51.24   99.27  5e-07   2600.4   51.61  100.00  5e-07
   268435456      67108864     float     sum   4891.6   54.88  106.32  5e-07   4821.8   55.67  107.86  5e-07
   536870912     134217728     float     sum   9393.7   57.15  110.73  5e-07   9318.5   57.61  111.63  5e-07
  1073741824     268435456     float     sum    18266   58.78  113.89  5e-07    18301   58.67  113.68  5e-07
  2147483648     536870912     float     sum    36391   59.01  114.34  5e-07    36710   58.50  113.34  5e-07
  4294967296    1073741824     float     sum    72501   59.24  114.78  5e-07    72344   59.37  115.03  5e-07
  8589934592    2147483648     float     sum   143927   59.68  115.64  5e-07   143961   59.67  115.61  5e-07
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 37.7108
#
```

Before running a new instance of a test, delete the old with `kubectl delete mpijob <job name>` or `kubectl delete mpijob --all`. Please note that it is important to wait for all pods from an earlier job to finish terminating before starting a new job with the same name.

## Running DeepSpeed Training Jobs
The MPI Operator can be used to run DeepSpeed based distributed training jobs similarly to how the NCCL test jobs are run. The MPI Operator creates the MPI hostsfile for you, and DeepSpeed can simply be run as a command like you would with a manual hostsfile setup.

## GDRCopy
[GDRCopy](https://developer.nvidia.com/gdrcopy) can be enabled to improve CPU to GPU memory communication in certain use cases. GDRCopy is supported in NCCL using a hidden environment variable `NCCL_GDRCOPY_ENABLE`. In our testing, performance improvements for regular NCCL allreduce workloads have not been measured. We do not recommend enabling GDRCopy for NCCL without performing adequate benchmarks to ensure that performance is improved. It is noted in the GDRCopy documentation that performance in some cases is degraded instead of improved.
