/*
 * nccl-tests build-time extension hooks.
 *
 * This header is the only contract between nccl-tests and an extension. nccl-tests
 * calls these functions from its existing output path when it is built with
 * -DNCCL_TESTS_EXTENSION, as described in ext_hooks.h in nccl-tests. Otherwise the calls
 * compile to nothing. Everything here is generic. It carries no product, host or schema names.
 *
 * Call order within the reporting process, which is rank 0 of an MPI job or the only process:
 *   init -> run_id -> run_id_shared -> identify_writer -> run_config -> device* ->
 *   (size_begin -> size_result{1,2})* -> footer -> process_end -> process_end_gathered ->
 *   error? -> finalize
 * Every other process of an MPI job calls init, run_id_shared, local_info, process_end and
 * finalize.
 * All hooks after init run on the main thread. An extension must never change the
 * test's stdout, stderr or exit code, and must never block for long.
 */
#ifndef NCCL_TESTS_EXT_H
#define NCCL_TESTS_EXT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Process start, right after MPI_Init. envp may be NULL. */
void nccl_tests_ext_init(int argc, char **argv, char **envp);

/* Called on rank 0 only, before the run id is broadcast. The extension may write its run id,
 * up to len - 1 bytes, into out. */
void nccl_tests_ext_run_id(char *out, size_t len);

/* Called on every process with the run id rank 0 wrote, after the broadcast. Builds without
 * MPI may not call it. */
void nccl_tests_ext_run_id_shared(const char *run_id);

/* Called with 1 by the thread that owns the report, the main thread of rank 0. Once set,
 * the writer stays the writer. Calls with 0 are ignored. */
void nccl_tests_ext_identify_writer(int is_writer);

/* The run's settings, as printed in the "# nThread ... " banner. */
void nccl_tests_ext_run_config(const char *program_name,
                               const char *nccl_tests_version,
                               int nccl_headers_code, int nccl_library_code,
                               int cuda_runtime_version,
                               int nthreads, int ngpus,
                               size_t min_bytes, size_t max_bytes,
                               size_t step_bytes, size_t step_factor,
                               int warmup_iters, int iters, int agg_iters,
                               int datacheck, int cuda_graph_launches,
                               int blocking_coll, int parallel_init);

/* Called on every process before the device records are gathered. The extension may write
 * up to len - 1 bytes of single-line text about this process's node into out. The text is
 * handed back to rank 0 with each device record of that process as local_info. */
void nccl_tests_ext_local_info(char *out, size_t len);

/* One participating GPU, as gathered to rank 0. serial, uuid and vbios may be "unknown" and
 * gpu_numa_node may be -1. The uuid is the NVML device UUID, which is unique per GPU where the
 * serial is shared per module. cpu_affinity and cpu_numa_nodes describe the process that
 * drives the GPU, as CPU and node lists such as "0-71" and "0". local_info is the text the
 * extension wrote for that process in nccl_tests_ext_local_info. */
void nccl_tests_ext_device(int rank, int group, int pid, const char *hostname,
                           int device, const char *pci_bus_id,
                           const char *device_name, const char *serial,
                           const char *uuid, const char *vbios, int gpu_numa_node,
                           const char *cpu_affinity, const char *cpu_numa_nodes,
                           const char *local_info);

/* Start of one result line, which covers one message size. */
void nccl_tests_ext_size_begin(size_t nbytes, size_t nelem,
                               const char *type_name, const char *op_name, int root);

/* One half of a result line. out_of_place is 1 for the first half and 0 for in-place.
 * cpu_time is 1 when time_usec is CPU time because the test runs with -C 1. */
void nccl_tests_ext_size_result(int out_of_place, double time_usec,
                                double alg_bw, double bus_bw,
                                int report_errors, long long wrong_elts,
                                int cpu_time);

/* Run summary. check_avg_bw is -1 when no minimum was requested. uuid may be NULL. */
void nccl_tests_ext_footer(int out_of_bounds, double avg_bus_bw,
                           double check_avg_bw, const char *uuid);

/* Called on every process at the end of the run, after the communicators are destroyed and
 * before MPI_Finalize. The extension may point data at up to len bytes it owns, which stay
 * valid until process_end_gathered returns on rank 0. */
void nccl_tests_ext_process_end(const char **data, size_t *len);

/* Called on rank 0 only, with what every process returned from process_end, indexed by
 * MPI rank. A build without MPI calls it with its own data and nprocs 1. */
void nccl_tests_ext_process_end_gathered(int nprocs, const char *const *data, const size_t *len);

/* Last NCCL error text, if any. May be NULL or empty. */
void nccl_tests_ext_error(const char *message);

/* End of run, on every process. result_code is the testResult_t that the process returns. */
void nccl_tests_ext_finalize(int result_code);

#ifdef __cplusplus
}
#endif

#endif /* NCCL_TESTS_EXT_H */
