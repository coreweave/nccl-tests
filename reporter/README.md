# nccl-tests-ext

The reporter that nccl-tests calls when it is built with an extension. It turns one
nccl-tests run into columnar rows of name, value, kind, version and context, in the NCCL
vocabulary of Gamble, the HPC verification suite, and PUTs them once to a configured results
endpoint. nccl-tests here means CoreWeave's fork of NVIDIA nccl-tests, `coreweave/nccl-tests-json`,
which carries the hook points. The design document lives outside this repository.

Status as of 2026-10-19. The library compiles and passes its GPU-free tests, on the host and
inside the official nccl-tests devel image. The fork branch builds with and without `EXT_DIR`,
with MPI on and off, and adds no compiler warnings. Hook symbols appear only in the `EXT_DIR`
binaries. Without a GPU, the stock and extension binaries are identical in stdout, stderr and
exit code apart from the PID, and the extension still reaches finalize on the CUDA-error exit
path. First GPU run on 2026-10-19 on one GB300 node with 4 GPUs, inside the nccl-tests container. Stock and extension built in 35 s each. `all_reduce_perf` ran with no measurable overhead and identical stdout structure, one PUT of 108 rows passed the API rules, and the error path still reached finalize. See the follow-ups at the end.

## Layout

- `include/nccl_tests_ext.h` is the hook contract and the only header nccl-tests includes.
  It is generic and carries no product, host or schema names, so it can live in the public
  fork too.
- `src/reporter.cc` holds the hooks, the in-memory state, the row building and the single
  send at finalize.
- `src/rows.*` holds the row model and the API limits. Keys are at most 64 characters, a row
  has at most 64 keys, strings are at most 32 KiB, reserved names are rejected and numbers must
  be finite. It also writes the compact JSON.
- `src/config.*` reads the environment and the mounted file. An empty URL means inactive.
- `src/http.*` makes one PUT through libcurl loaded with `dlopen`. No curl headers are needed
  to build. A missing libcurl at run time means not delivered, never a crash.
- `src/util.*` has UUIDv7, ISO timestamps, size labels and version formatting.
- `test/` has `fake_run.cc`, which drives the hooks like `all_reduce_perf` does,
  `mock_endpoint.py`, which validates rows against the API rules, and `run_tests.sh`, which runs
  the scenarios.

## Build

    make            # build/libnccltestsext.a
    make test       # harness and scenarios, no GPU or network needed

From the nccl-tests branch with the hook points:

    make -C src EXT_DIR=/path/to/nccl-tests-ext [MPI=1 ...]

`EXT_DIR` adds `-DNCCL_TESTS_EXTENSION -I$(EXT_DIR)/include` and links the library and
`-ldl`. Without `EXT_DIR` the fork builds exactly as before and every hook is an empty
statement. Build this library with the same compiler family as the nccl-tests host
compiler, the one given to `-ccbin`, since it is linked into the nvcc link step.

## Compile check without a GPU

    tools/build-in-image.sh [image] [fork-dir]

This pulls the official `nccl-tests` devel image, which has nvcc, NCCL and HPC-X MPI, builds
this library and runs its tests with the image's compiler, then builds the fork four ways
into `build/image-build/`. The four ways are stock and `EXT_DIR`, each with MPI=0 and MPI=1,
for one GPU architecture by default. Compiling needs no GPU. Only running does. The image
ships `libcurl.so.4`, so the `dlopen` path works there.

## Hook points in nccl-tests

The fork's `src/ext_hooks.h` defines the macros that wrap each call.

| Hook | Where | Carries |
|---|---|---|
| `nccl_tests_ext_init` | `main()`, after `MPI_Init` | argv, env, start time |
| `nccl_tests_ext_identify_writer` | `run()`, where `is_main_thread` is set | the rank 0 main thread is the writer |
| `nccl_tests_ext_run_config` | `writeDeviceReport` | versions, nThreads, nGpus, sizes, iterations, flags |
| `nccl_tests_ext_device` | the device record loop, which now runs with or without `-J` | rank, host, device, PCI id, name, serial |
| `nccl_tests_ext_size_begin` and `_size_result` | `writeBenchmarkLinePreamble` and `writeBenchmarkLineBody` | size, count, type, op, root, then time, alg and bus bandwidth, wrong elements |
| `nccl_tests_ext_footer` | `writeResultFooter` | out-of-bounds count, average bus bandwidth, minimum-bandwidth check, run uuid |
| `nccl_tests_ext_error` | `writeErrors` | last NCCL error text |
| `nccl_tests_ext_finalize` | `main()`, after `run()` returns | result code, then builds and sends |

Two small changes to the fork go beyond the hook lines. The device-record parse loop runs
when `write_json || NCCL_TESTS_EXT_ENABLED`, while JSON output is still written only under
`write_json`. The run uuid is generated before the `write_json` test in the footer so that
JSON and the extension share one id.

## Configuration

Everything comes from the environment. Nothing is compiled in.

| Variable | Purpose |
|---|---|
| `NCCL_TESTS_REPORT_URL` | Endpoint URL, may contain `{run_id}`. Set but empty means disabled. |
| `NCCL_TESTS_REPORT_URL_FILE` | File to read the URL from when the variable is unset. Default `/etc/perf-telemetry/url`. |
| `NCCL_TESTS_REPORT=0` | Opt out. |
| `NCCL_TESTS_REPORT_RUN_ID` | Override the run id. The default is a new UUIDv7 per run. Cascade, the collector, keeps only the last write per run id and kind. |
| `NCCL_TESTS_REPORT_AUTH_HEADER` | One header line, sent verbatim. |
| `NCCL_TESTS_REPORT_CERT`, `_KEY`, `_CA` | Client certificate, key and CA bundle for the mTLS endpoint. |
| `NCCL_TESTS_REPORT_IMAGE` | Image name or tag to record. |
| `NCCL_TESTS_REPORT_TAG_<NAME>` | Recorded as context `tag_<name>`. |
| `NCCL_TESTS_REPORT_CONNECT_TIMEOUT`, `_MAX_TIME` | Seconds. Defaults 2 and 5. |
| `NCCL_TESTS_REPORT_DEBUG=1` | Log to stderr. Otherwise silent. |

Behavior. The reporter never changes stdout, stderr or the exit code. It makes one attempt
with short timeouts and no retries. Only the writer rank sends. No URL means no work at all.

## Rows, version 1.0.0

- `nccl-test-metric` has, per message size, `bandwidth` as a sample in GB/s of out-of-place bus
  bandwidth, `latency` as a mean in us, and `wrong_elements` as a sample count with both halves
  summed. Per run it has `bandwidth` as a mean in GB/s with a `verdict`, `returncode` and
  `duration` in seconds. Context carries `message_size`, `size_bytes`, `count`, `dtype`,
  `redop` and `root`, plus the common keys `node`, `test_name`, `topology=collective`,
  `nccl_version`, `nccl_library_version`, `nccl_headers_version`, `nccl_tests_version`,
  `cuda_version`, `image`, `job_id`, `uuid`, `started_at`, `ended_at` and `tag_*`.
- `nccl-test-config` has every `NCCL_*` variable with `source=environment`, the test's settings
  with `source=test_config`, `argv` with `source=command_line` and `error_message` with
  `source=errors`.
- `nccl-test-node-facts` has one `gpu` row per GPU. The value is the serial, else the PCI bus
  id. Context carries `node`, `gpu`, `serial`, `pci_bus_id`, `device_name`, `rank` and `group`.

Two differences from the shim script that preceded this reporter. `nccl_headers_version` is
available natively and is emitted. `verdict` comes from the run's result code rather than from
parsed JSON.

## Follow-ups

1. Multi-node run with MPI, then the real collector. The single-node GPU run is done.
2. Device UUID. The gathered device record carries the serial only, and on GB300 the NVML serial is per
   module, so two GPUs share one serial. Per-GPU identity needs the UUID, which means a change to the record
   format in the fork.
3. `--no-report` as a command-line flag. Today the opt-out is `NCCL_TESTS_REPORT=0`.
4. The mTLS path against the real collector, and which clusters have which endpoint.
5. Exclude or dedupe `hpc-verification` runs, which Gamble already reports.
