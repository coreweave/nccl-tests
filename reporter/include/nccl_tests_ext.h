/*
 * nccl-tests build-time extension hooks.
 *
 * This header is the only contract between nccl-tests and an extension. nccl-tests
 * calls these functions from its existing output path when it is built with
 * -DNCCL_TESTS_EXTENSION, as described in ext_hooks.h in nccl-tests. Otherwise the calls
 * compile to nothing. Everything here is generic. It carries no product, host or schema names.
 *
 * Call order within the reporting process, which is rank 0 of an MPI job or the only process:
 *   init -> identify_writer -> run_config -> device* -> (size_begin -> size_result{1,2})* -> footer -> error? -> finalize
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

/* One participating GPU, as gathered to rank 0. serial and uuid may be "unknown". The uuid
 * is the NVML device UUID, which is unique per GPU where the serial is shared per module. */
void nccl_tests_ext_device(int rank, int group, int pid, const char *hostname,
                           int device, const char *pci_bus_id,
                           const char *device_name, const char *serial,
                           const char *uuid);

/* Start of one result line, which covers one message size. */
void nccl_tests_ext_size_begin(size_t nbytes, size_t nelem,
                               const char *type_name, const char *op_name, int root);

/* One half of a result line. out_of_place is 1 for the first half and 0 for in-place. */
void nccl_tests_ext_size_result(int out_of_place, double time_usec,
                                double alg_bw, double bus_bw,
                                int report_errors, long long wrong_elts);

/* Run summary. check_avg_bw is -1 when no minimum was requested. uuid may be NULL. */
void nccl_tests_ext_footer(int out_of_bounds, double avg_bus_bw,
                           double check_avg_bw, const char *uuid);

/* Last NCCL error text, if any. May be NULL or empty. */
void nccl_tests_ext_error(const char *message);

/* End of run. result_code is the testResult_t that the process returns. */
void nccl_tests_ext_finalize(int result_code);

#ifdef __cplusplus
}
#endif

#endif /* NCCL_TESTS_EXT_H */
