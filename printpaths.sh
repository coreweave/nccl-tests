#!/bin/bash
# Prints a list of currently-set HPCX variables in the format `[prefix ]NAME=value`

prefix=${1:+$1 }

for var in \
  HPCX_DIR \
  HPCX_UCX_DIR \
  HPCX_UCC_DIR \
  HPCX_SHARP_DIR \
  HPCX_NCCL_RDMA_SHARP_PLUGIN_DIR \
  HPCX_HCOLL_DIR \
  HPCX_MPI_DIR \
  HPCX_OSHMEM_DIR \
  HPCX_MPI_TESTS_DIR \
  HPCX_OSU_DIR \
  HPCX_OSU_CUDA_DIR \
  HPCX_IPM_DIR \
  HPCX_CLUSTERKIT_DIR \
  OMPI_HOME \
  MPI_HOME \
  OSHMEM_HOME \
  OPAL_PREFIX \
  OLD_PATH \
  PATH \
  OLD_LD_LIBRARY_PATH \
  LD_LIBRARY_PATH \
  OLD_LIBRARY_PATH \
  LIBRARY_PATH \
  OLD_CPATH \
  CPATH \
  PKG_CONFIG_PATH
do
  # Prefix + VAR=$VAR or VAR="" if $VAR is empty
  echo "${prefix}${var}=${!var:-\"\"}"
done