// A stand-in for an nccl-tests binary. It calls the extension hooks in the same order
// and with the same shapes as all_reduce_perf would, then exits with FAKE_RC.
// No GPU, NCCL or MPI involved. Environment knobs, with defaults:
//   FAKE_RC is the exit and result code, default 0. FAKE_WRITER default 1. FAKE_NDEV default 4.
//   FAKE_NSIZES default 10. FAKE_ERR is an NCCL error text. FAKE_WRONG, when set, gives
//   size 3 three wrong elements. FAKE_CPUTIME, when set, reports the times as CPU time.
//   FAKE_CYCLES repeats the sweep, as -N does. FAKE_OOB and FAKE_CHECK set the footer counts.
#include <stdio.h>
#include <stdlib.h>

#include "nccl_tests_ext.h"

static int envi(const char *n, int d) { const char *v = getenv(n); return v && *v ? atoi(v) : d; }

int main(int argc, char **argv, char **envp) {
  int rc = envi("FAKE_RC", 0), writer = envi("FAKE_WRITER", 1);
  int ndev = envi("FAKE_NDEV", 4), nsizes = envi("FAKE_NSIZES", 10);

  nccl_tests_ext_init(argc, argv, envp);
  printf("# nccl-tests version 2.18.3 nccl-headers=22104 nccl-library=23102\n");
  printf("# Collective test starting: all_reduce_perf\n");
  nccl_tests_ext_identify_writer(writer);
  nccl_tests_ext_run_config("all_reduce_perf", "2.18.3", 22104, 23102, 12040,
                            1, ndev, 8, 1ULL << 30, 0, 2, 5, 20, 1, 1, 0, 0, 0);
  // Every process contributes its local info before the gather, as the fork does.
  char local[1024];
  nccl_tests_ext_local_info(local, sizeof(local));
  for (int i = 0; i < ndev; ++i) {
    char pci[32]; snprintf(pci, sizeof(pci), "0000:%02x:00", 0x1b + i);
    // Device 0 has a uuid only, 1 a serial only, 2 neither, 3 both. Devices 1 and 3 share a
    // serial, as two GPUs of one GB300 module do.
    char uuid[48]; snprintf(uuid, sizeof(uuid), "GPU-663051f8-43a9-e63c-6886-da4b882cd7%02x", i);
    nccl_tests_ext_device(i, 0, 4242, i < 2 ? "fakehost-01" : "fakehost-02", i, pci, "NVIDIA GB300 ",
                          (i % 2) ? "1650123456789" : "unknown",
                          (i == 0 || i == 3) ? uuid : "unknown",
                          i == 0 ? "97.00.D0.00.5B" : "unknown", i < 2 ? 0 : 1,
                          i < 2 ? "0-71" : "72-143", i < 2 ? "0" : "1", local);
  }
  int cycles = envi("FAKE_CYCLES", 1);
  for (int cyc = 0; cyc < cycles; ++cyc) {
  unsigned long long size = 8;
  for (int i = 0; i < nsizes; ++i, size *= 2) {
    nccl_tests_ext_size_begin(size, size / 4, "float", "sum", -1);
    int cpu = getenv("FAKE_CPUTIME") ? 1 : 0;
    nccl_tests_ext_size_result(1, 12.5 + i, size / 1e3, 100.0 + i, 1, 0, cpu);
    nccl_tests_ext_size_result(0, 12.0 + i, size / 1e3, 101.0 + i, 1, (i == 3 && getenv("FAKE_WRONG")) ? 3 : 0, cpu);
  }
  }
  nccl_tests_ext_footer(envi("FAKE_OOB", 0), 150.25, getenv("FAKE_CHECK") ? atof(getenv("FAKE_CHECK")) : -1,
                        "11111111-2222-4333-8444-555555555555");
  if (getenv("FAKE_ERR")) nccl_tests_ext_error(getenv("FAKE_ERR"));
  printf("# Collective test concluded: all_reduce_perf\n");
  nccl_tests_ext_finalize(rc);
  return rc;
}
