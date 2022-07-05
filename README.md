# NCCL for Distributed Training

CoreWeave supports the [NVIDIA Collective Communication Library (NCCL)](https://developer.nvidia.com/nccl) for powering multi GPU and multi node nueral network training. NCCL underpins the vast majority of all distributed training frameworks such as [DeepSpeed](https://github.com/microsoft/DeepSpeed), [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html) and [Horovod](https://horovod.readthedocs.io/en/stable/gpus_include.html).

NCCL is supported across all CoreWeave NVIDIA GPUs over Ethernet. In addition, the specialized A100 HGX clusters are built to the design of NVIDIA DGX SuperPODs, including [NVIDIA Quantum InfiniBand](https://www.nvidia.com/en-us/networking/quantum2/) networking and in-network collections using [NVIDIA SHARP](https://docs.nvidia.com/networking/display/SHARPv270/Introduction) to deliver the highest distributed training performance possible.

## Docker Images
This repository includes Dockerfiles that can be used directly or as a template for your distributed training appictions. The Dockerfiles include the following components:
- NVIDIA [Mellanox OFED Driver](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/) userspace components. The kernel side is installed on our bare-metal nodes and do not need to be installed by users. The OFED drivers are necessary for optimized InfiniBand communication.
- NVIDIA [HPC-X](https://developer.nvidia.com/networking/hpc-x) which is a packaging of OpenMPI and UCX
- NVIDIA [GDRCopy](https://developer.nvidia.com/gdrcopy) libraries leverages GPUDirect RDMA for improved GPU to host memory copy performance in certain applications. The kernel support for GDRCopy exists on CoreWeaves bare-metal nodes. GDRCopy is only supported on A100 training clusters.
- NVIDIA [NCCL SHARP Plugin](https://github.com/Mellanox/nccl-rdma-sharp-plugins) for SHARP support in NCCL
- NVIDIA [NCCL Tests](https://github.com/NVIDIA/nccl-tests) for verification purposes
- OpenSSH server and related settings to enable images to easily be used as MPI Runners

CoreWeave also [publishes images](https://hub.docker.com/r/coreweave/nccl-tests/tags) built from these Dockerfiles that can be used as base for your own images. The newest image at time of writing is `coreweave/nccl-tests:2022-07-05_09-54-20.051_EDT` built in CUDA 11.6.2 with HPC-X 2.11.

## Running NCCL Tests
CoreWeave provides a managed instance of the [MPI Operator](https://github.com/kubeflow/mpi-operator) to allow running MPI Jobs in a container native fashion. No installation is required by the user, simply execute a MPIJob manifest in your namespace.

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
#                                                       out-of-place                       in-place          
#       size         count      type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           4             1     float     sum    58.20    0.00    0.00  5e-07    58.00    0.00    0.00  7e-07
           8             2     float     sum    57.89    0.00    0.00  5e-07    57.63    0.00    0.00  7e-07
          16             4     float     sum    58.47    0.00    0.00  7e-07    57.69    0.00    0.00  7e-07
          32             8     float     sum    64.56    0.00    0.00  1e-06    59.70    0.00    0.00  1e-06
          64            16     float     sum    58.65    0.00    0.00  1e-06    58.84    0.00    0.00  1e-06
         128            32     float     sum    59.34    0.00    0.00  1e-06    59.26    0.00    0.00  1e-06
         256            64     float     sum    61.66    0.00    0.01  1e-06    61.68    0.00    0.01  1e-06
         512           128     float     sum    69.09    0.01    0.01  5e-07    68.09    0.01    0.01  5e-07
        1024           256     float     sum    73.64    0.01    0.03  1e-06    72.76    0.01    0.03  1e-06
        2048           512     float     sum    81.99    0.02    0.05  1e-06    81.50    0.03    0.05  1e-06
        4096          1024     float     sum    86.23    0.05    0.09  1e-06    84.49    0.05    0.10  1e-06
        8192          2048     float     sum    91.92    0.09    0.18  1e-06    87.92    0.09    0.18  1e-06
       16384          4096     float     sum    94.72    0.17    0.34  1e-06    89.89    0.18    0.36  1e-06
       32768          8192     float     sum    102.5    0.32    0.63  1e-06    94.79    0.35    0.69  1e-06
       65536         16384     float     sum    120.5    0.54    1.08  1e-06    117.4    0.56    1.11  1e-06
      131072         32768     float     sum    141.3    0.93    1.84  1e-06    140.0    0.94    1.86  1e-06
      262144         65536     float     sum    150.2    1.75    3.46  1e-06    148.8    1.76    3.50  1e-06
      524288        131072     float     sum    168.7    3.11    6.17  1e-06    168.7    3.11    6.17  1e-06
     1048576        262144     float     sum    232.5    4.51    8.95  1e-06    212.7    4.93    9.78  1e-06
     2097152        524288     float     sum    294.1    7.13   14.15  1e-06    292.1    7.18   14.25  1e-06
     4194304       1048576     float     sum    421.4    9.95   19.75  1e-06    427.6    9.81   19.47  1e-06
     8388608       2097152     float     sum    635.4   13.20   26.20  1e-06    633.3   13.25   26.29  1e-06
    16777216       4194304     float     sum   1081.3   15.52   30.79  1e-06   1102.7   15.22   30.19  1e-06
    33554432       8388608     float     sum   1982.7   16.92   33.58  1e-06   1995.8   16.81   33.36  1e-06
    67108864      16777216     float     sum   3931.7   17.07   33.87  2e-06   4036.1   16.63   32.99  2e-06
   134217728      33554432     float     sum   7141.1   18.80   37.30  1e-06   7073.1   18.98   37.65  1e-06
   268435456      67108864     float     sum    12737   21.08   41.82  2e-06    12811   20.95   41.58  2e-06
   536870912     134217728     float     sum    25775   20.83   41.33  2e-06    25779   20.83   41.33  2e-06
  1073741824     268435456     float     sum    46911   22.89   45.42  2e-06    46889   22.90   45.44  2e-06
  2147483648     536870912     float     sum    95238   22.55   44.74  2e-06    95289   22.54   44.72  2e-06
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 13.0489 
#
```

Before running a new instance of a test, delete the old with `kubectl delete mpijob <job name>` or `kubectl delete mpijob --all`. Please note that it is important to wait for all pods from an earlier job to finish terminating before starting a new job with the same name.

## Running DeepSpeed Training Jobs
The MPI Operator can be used to run DeepSpeed based distributed training jobs similarly to how the NCCL test jobs are run. The MPI Operator creates the MPI hostsfile for you, and DeepSpeed can simply be run as command like you would with a manual hostsfile setup.

## GDRCopy
[GDRCopy](https://developer.nvidia.com/gdrcopy) can be enabled to improve CPU to GPU memory communication in certain use cases. GDRCopy is supported in NCCL using a hidden environment variable `NCCL_GDRCOPY_ENABLE`. In our testing, performance improvements for regular NCCL allreduce workloads have not been measured. We do not recommend enabling GDRCopy for NCCL without performing adequate benchmarks to ensure that performance is improved. It is noted in the GDRCopy documentation that performance in some cases is degraded instead of improved.
