# Example NCCL Test Scripts

These Slurm sbatch scripts are good starting points to use for submission scripts for your own training jobs.

## Basic Usage

Use a partition corresponding to one specific type of GPU (e.g. h100, h200). Depending on your cluster, the default
partition may include different GPUs that will be on different InfiniBand fabrics. See what partitions you have
using `sinfo`:

```sh
sinfo
```

Then submit your sbatch script directly to Slurm, either replacing `$PARTITION` below with the name
of the GPU partition you want to use, or setting it as a variable.

Note also that the scripts are configured by default to use 8 nodes. If you have more or less nodes,
override the number of nodes by using the `-N` parameter (below we use 4 nodes).

```sh
sbatch --partition="$PARTITION" -N 4 nccl-test-hgx-ib.slurm
```

The scripts are grouped by SKU family rather than a specific GPU model:

- `nccl-test-hgx-ib*.slurm` — standard 8-GPU HGX SKUs with InfiniBand
  (e.g. gd-8xh100ib-i128, gd-8xh200ib-i128, gd-8xb200ib-i128). Pick the
  bare-metal variant, the enroot variant, or the SHARP enroot variant.
- `nccl-test-nvl-ib-enroot.slurm` — rack-scale NVL SKUs with InfiniBand
  (e.g. gb200-4x).
- `nccl-test-nvl-roce-enroot.slurm` — rack-scale NVL SKUs with RoCE
  (e.g. gb300-4x).

The output will be put into a file of the form `nccl_test_allreduce_jobID.out`.
You can check on job progress using the command:

```sh
tail -f nccl_test_allreduce_*.out
```

## Using as Examples

There are a few practices illustrated by these sbatch scripts that we recommend adopting.

- Organize job output by job ID. If there is an error, we can use the job ID to quickly examine the metrics associated with the job.
- Pull the container first and then use it from a shared directory. This will speed everything up considerably.
- Use the environment variables here to ensure optimal performance.
