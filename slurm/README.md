# Example NCCL Test Scripts

These slurm sbatch scripts are good starting points to use for submission scripts for your own training jobs.

## Basic Usage

Use a partition corresponding to one specific type of gpu (e.g. h100, h200). Depending on your cluster, the default partition may include different gpus that will be on different infiniband fabrics. See what partitions you have using sinfo:

```sh
sinfo
```

Then submit your sbatch script directly to Slurm, either replacing `$PARTITION` below with the name of the gpu partition you want to use, or setting it as a variable.

Note also that the scripts are configured by default to use 8 nodes. If you have more or less nodes, override the number of nodes by using the `-N` parameter (below we use 4 nodes).

```sh
sbatch --partition="$PARTITION" -N 4 nccl-test-distributed-h100-64.slurm
```

The output will be put into a file of the form `nccl_test_allreduce_jobID.out`. You can check on job progress using the command:

```sh
tail -f nccl_test_allreduce_*.out
```

## Using as Examples

There are a few practices illustrated by these sbatch scripts that we recommend adopting.

- Organize job output by job ID. If there is an error, we can use the job ID to quickly examine the metrics associated with the job.
- Pull the container first and then use it from a shared directory. This will speed everything up considerably.
- Use the environment variables here to ensure best performance.