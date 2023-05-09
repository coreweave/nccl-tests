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

| **Image Tag**                                                              | **CUDA** | **NCCL** | **HPC-X** |
|----------------------------------------------------------------------------|----------|----------|-----------|
| ghcr.io/coreweave/nccl-tests:12.1.1-devel-ubuntu20.04-nccl2.18.1-1-TODO    | 12.1.1   | 2.18.1   | 2.15.0    |
| ghcr.io/coreweave/nccl-tests:12.0.1-devel-ubuntu20.04-nccl2.18.1-1-TODO    | 12.0.1   | 2.18.1   | 2.15.0    |
| ghcr.io/coreweave/nccl-tests:11.8.0-devel-ubuntu20.04-nccl2.16.2-1-4a46534 | 11.8.0   | 2.16.2   | 2.14.0    |
| ghcr.io/coreweave/nccl-tests:11.7.1-devel-ubuntu20.04-nccl2.14.3-1-4a46534 | 11.7.1   | 2.14.3   | 2.14.0    |
| coreweave/nccl-tests:2022-09-28_16-34-19.392_EDT                           | 11.6.2   | 2.12.0   | 2.12      |

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
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
   536870912     134217728     float     sum      -1   7035.5   76.31  147.85      0   7011.8   76.57  148.35      0
  1073741824     268435456     float     sum      -1    13639   78.73  152.53      0    13418   80.02  155.04      0
  2147483648     536870912     float     sum      -1    26476   81.11  157.15      0    26677   80.50  155.97      0
  4294967296    1073741824     float     sum      -1    52553   81.73  158.34      0    52502   81.81  158.50      0
  8589934592    2147483648     float     sum      -1   104180   82.45  159.75      0   106210   80.88  156.70      0
```

Before running a new instance of a test, delete the old with `kubectl delete mpijob <job name>` or `kubectl delete mpijob --all`. Please note that it is important to wait for all pods from an earlier job to finish terminating before starting a new job with the same name.

## Running DeepSpeed Training Jobs
The MPI Operator can be used to run DeepSpeed based distributed training jobs similarly to how the NCCL test jobs are run. The MPI Operator creates the MPI hostsfile for you, and DeepSpeed can simply be run as a command like you would with a manual hostsfile setup.

## GDRCopy
[GDRCopy](https://developer.nvidia.com/gdrcopy) can be enabled to improve CPU to GPU memory communication in certain use cases. GDRCopy is supported in NCCL using a hidden environment variable `NCCL_GDRCOPY_ENABLE`. In our testing, performance improvements for regular NCCL allreduce workloads have not been measured. We do not recommend enabling GDRCopy for NCCL without performing adequate benchmarks to ensure that performance is improved. It is noted in the GDRCopy documentation that performance in some cases is degraded instead of improved.
