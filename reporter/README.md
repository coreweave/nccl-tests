# nccl-tests-ext

The reporter that nccl-tests calls when it is built with an extension. It turns one
nccl-tests run into columnar rows of name, value, kind, version and context, and PUTs them once to a results
collector. nccl-tests here
means CoreWeave's fork of NVIDIA nccl-tests, `coreweave/nccl-tests-json`, which carries the hook
points.

The library compiles and passes its GPU-free tests on the host and inside the official
nccl-tests devel image. The fork builds with and without `EXT_DIR`, with MPI on and off, and adds
no compiler warnings. Hook symbols appear only in the `EXT_DIR` binaries. Without a GPU, the stock
and extension binaries are identical in stdout, stderr and exit code apart from the PID, and the
extension still reaches finalize on the CUDA-error exit path. On a GPU node the extension adds no
measurable overhead and leaves the stdout structure unchanged.

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
| `nccl_tests_ext_run_id` and `_run_id_shared` | `run()`, after the host hash gather | rank 0 writes its run id, `MPI_Bcast` hands it to every process |
| `nccl_tests_ext_identify_writer` | `run()`, where `is_main_thread` is set | the rank 0 main thread is the writer |
| `nccl_tests_ext_run_config` | `writeDeviceReport` | versions, nThreads, nGpus, sizes, iterations, flags |
| `nccl_tests_ext_local_info` | the device record loop, on every process | a line of text the extension writes about its node |
| `nccl_tests_ext_device` | the device record loop, which now runs with or without `-J` | rank, host, device, PCI id, name, serial, device UUID, VBIOS, GPU NUMA node, process CPU affinity and NUMA nodes, local info |
| `nccl_tests_ext_size_begin` and `_size_result` | `writeBenchmarkLinePreamble` and `writeBenchmarkLineBody` | size, count, type, op, root, then time, alg and bus bandwidth, wrong elements and whether the time is CPU time |
| `nccl_tests_ext_process_end` and `_process_end_gathered` | `run()`, after the footer and before `MPI_Finalize`, on every process | each process's end-of-run text, gathered to rank 0 with `MPI_Gatherv` |
| `nccl_tests_ext_footer` | `writeResultFooter` | out-of-bounds count, average bus bandwidth, minimum-bandwidth check, run uuid |
| `nccl_tests_ext_error` | `writeErrors` | last NCCL error text |
| `nccl_tests_ext_finalize` | `main()`, after `run()` returns, on every process | result code, then the writer builds and sends, and each process uploads its logs when asked |

Two small changes to the fork go beyond the hook lines. The device-record parse loop runs
when `write_json || NCCL_TESTS_EXT_ENABLED`, while JSON output is still written only under
`write_json`. The run uuid is generated before the `write_json` test in the footer so that
JSON and the extension share one id.

## Activation

In order of precedence. `NCCL_TESTS_REPORT=0` turns the reporter off. `NCCL_TESTS_REPORT=1` forces it
on. A configured URL turns it on. A pod namespace on the skip list turns it off. Inside a cluster,
which means `resolv.conf` has a `cluster.local` search domain, the reporter tries the candidate
collector hosts in order and uses the first that resolves, on port 4318. Otherwise
it is off. Resolution happens at finalize, on the writer rank only, so the measured run is untouched
and a large MPI job does not query DNS from every rank. A build without candidate hosts never reports
unless the environment configures an endpoint.

The candidate hosts and the skip list are build-time defaults, `make lib DEFAULT_HOSTS=a,b
SKIP_NAMESPACES=x,y`. The image build sets them. The pod namespace comes from the service account
namespace file, else from the first `resolv.conf` search domain, which is `<namespace>.svc.cluster.local`
in every Kubernetes pod.

## Configuration

Environment variables override the build-time defaults.

| Variable | Purpose |
|---|---|
| `NCCL_TESTS_REPORT_URL` | Endpoint URL, may contain `{run_id}`. Set but empty means disabled. |
| `NCCL_TESTS_REPORT_URL_FILE` | File to read the URL from when the variable is unset. Default `/etc/perf-telemetry/url`. |
| `NCCL_TESTS_REPORT=0` | Opt out. |
| `NCCL_TESTS_REPORT=1` | Force on, even outside a cluster or in a skipped namespace. |
| `NCCL_TESTS_REPORT_HOSTS` | Candidate collector hosts, comma separated, tried in order. |
| `NCCL_TESTS_REPORT_PORT` | Port for the candidate hosts. Default 4318. |
| `NCCL_TESTS_REPORT_SKIP_NAMESPACES` | Pod namespaces that never report, comma separated. |
| `NCCL_TESTS_REPORT_NAMESPACE` | Override the discovered pod namespace. |
| `NCCL_TESTS_REPORT_NAMESPACE_FILE`, `_RESOLV_CONF` | Alternate paths for the namespace file and `resolv.conf`, for tests. |
| `NCCL_TESTS_REPORT_RUN_ID` | Override the run id. The default is a new UUIDv7 per run. |
| `NCCL_TESTS_REPORT_AUTH_HEADER` | One header line, sent verbatim. |
| `NCCL_TESTS_REPORT_IMAGE` | Image name or tag to record. The image build sets it to the image's own reference, so every report names the exact build. |
| `NCCL_TESTS_REPORT_TAG_<NAME>` | Recorded as context `tag_<name>`. |
| `NCCL_TESTS_REPORT_CONNECT_TIMEOUT`, `_MAX_TIME` | Seconds. Defaults 2 and 5. |
| `NCCL_TESTS_REPORT_DEBUG=1` | Log to stderr. Otherwise silent. |
| `NCCL_TESTS_REPORT_LOGS=1` | Also send each process's NCCL debug log. Off by default. See below. |
| `NCCL_TESTS_REPORT_ARTIFACT_URL` | Artifact endpoint URL with `{run_id}` and `{file}`. |
| `NCCL_TESTS_REPORT_ARTIFACT_HOSTS` | Candidate artifact hosts, comma separated, tried in order inside a cluster. |
| `NCCL_TESTS_REPORT_LOG_MAX_MIB` | Largest log for the artifact endpoint, in MiB. Default 64, at most 256. A larger one goes with the results. |
| `NCCL_TESTS_REPORT_LOG_MAX_TIME` | Seconds per upload. Default 30. |
| `NCCL_TESTS_REPORT_LOG_ROWS_MAX_KIB` | Largest log per process sent with the results, in KiB. Default 1024, at most 8192. |

Behavior. The reporter never changes stdout, stderr or the exit code. It makes one attempt
with short timeouts and no retries. Only the writer rank sends. No URL means no work at all.
Name resolution of the candidate hosts shares one deadline, the connect timeout. Zero
timeouts fall back to the defaults. A run that produces more than 4096 result lines, such as
`-N 0`, stops recording and sends nothing.

## Log upload

With `NCCL_TESTS_REPORT_LOGS=1` on an active reporter, every process sends the NCCL debug log
the user already asked NCCL to write, `NCCL_DEBUG_FILE` with `%h`, `%p` and `%%` expanded as
NCCL does. NCCL writes that file only when `NCCL_DEBUG` is `WARN` or more verbose. Nothing
changes where NCCL or the test writes its output, and no other file is read. The `-J` output
file is never sent, since it holds the whole environment of the process.

The log needs nothing beyond the results endpoint.

- When an artifact endpoint is reachable, each process PUTs its log to
  `/v1/runs/{run_id}/artifacts/nccl-debug-<host>-<pid>.log` under the run id of the results.
  The endpoint is `NCCL_TESTS_REPORT_ARTIFACT_URL` or else the first of the artifact hosts that
  resolves inside a cluster, on the reporter port.
- Otherwise, or when that upload fails or the file is over the artifact size limit, the process
  hands the log to rank 0 at the end of the run, through the fork's end-of-run gather. Rank 0
  sends those logs to the results endpoint in a second PUT of kind `nccl-test-log`, one row per
  piece of up to 30000 bytes with `node`, `pid`, `file`, `part`, `parts`, `bytes` and
  `truncated` in context. A log over `NCCL_TESTS_REPORT_LOG_ROWS_MAX_KIB`, default 1024, keeps
  its first and last parts, cut at character boundaries. Each process also keeps to its share
  of 12 MiB across all processes, so all logs together fit one request.
- Rank 0 records one `log_status` config row per process with `source=logs`, so the outcome of
  every node is in the results. The value is `delivered`, `in_results`, `no_debug_file`,
  `missing`, `unreadable` or `over_budget`, and context has `log_node`, `pid`, `file`, `bytes` and, when cut,
  `truncated`.

Rank 0 shares its run id with every process right after MPI init. A process without the shared
run id, or with a results URL that has no `{run_id}` and no `NCCL_TESTS_REPORT_RUN_ID`, skips the
artifact endpoint and hands its log to rank 0. A failed upload never changes the test.

## Recorded environment variables

The reporter reads a variable only when its name starts with one of the prefixes below, and
sends it only when it is set. These prefixes cover the libraries that decide how NCCL moves
data. Everything else in the environment, such as the shell, the scheduler and anything a user
adds, is never read.

| Stack | Prefixes |
|---|---|
| NCCL | `NCCL_` |
| UCX and UCC | `UCX_`, `UCC_` |
| HCOLL and SHARP | `HCOLL_`, `SHARP_` |
| MPI runtimes | `OMPI_`, `OPAL_`, `PRTE_`, `PMIX_`, `PMI_`, `MPICH_`, `I_MPI_` |
| CUDA | `CUDA_` |
| RDMA | `GDRCOPY_`, `MLX5_`, `RDMAV_`, `IBV_` |

Three rules keep credentials out, and they follow the environment filter of the fork's JSON
output. A name that contains KEY, TOKEN, SECRET, PASS, PWD, CRED, AUTH, PRIVATE, SIGNING,
SIGNATURE, HMAC, SALT, JWT, BEARER, COOKIE, SESSION, LICENSE, LICENCE, DSN, URL, DATABASE,
REDIS, MONGODB, CONNECTION_STRING, USERNAME, USER or EMAIL is dropped, except that a name
containing PKEY stays, because InfiniBand partition keys are settings. A value that starts
like a credential from a known service, for example `Bearer ` or `ghp_`, is sent as
`<redacted>`, so the variable is known to exist without its content. The reporter's own
`NCCL_TESTS_REPORT_*` settings are never sent as config rows. Names are compared without
regard to case. Other values, including file paths, addresses and interface names, are sent
as they are.

The `nccl_version` context is the version of the NCCL library loaded in the process, never the
`NCCL_VERSION` variable, which a base image may set for an NCCL that the image no longer ships.
The reporter also reads `SLURM_JOB_ID` into `job_id`, `HPCX_DIR` only to find the HPC-X version
file, and its own `NCCL_TESTS_REPORT_*` settings. Of those settings only the values of
`NCCL_TESTS_REPORT_TAG_<NAME>` and `NCCL_TESTS_REPORT_IMAGE` are sent, as `tag_<name>` and `image`.

## Rows, version 1.0.0

- `nccl-test-metric` has, per message size and for both halves of the result line, `bandwidth`
  as a sample in GB/s of bus bandwidth, `alg_bandwidth` as a sample of algorithm bandwidth and
  `latency` as a mean in us, with `_in_place` appended for the in-place half. Under `-C 1` the
  time is CPU time and is sent as `cpu_time` instead of `latency`. `wrong_elements` is a sample
  count with both halves summed. Per run it has `bandwidth` as a mean in GB/s with a `verdict`,
  `returncode`, `duration` in seconds, `out_of_bounds` and, with `-a`, `min_bandwidth_check`.
  With `-N` above 1 every per-size row carries `cycle`, counted from 0 within its data type, op
  and root, and a `cycles` row gives the number of cycles. Context carries `message_size`, `size_bytes`, `count`, `dtype`,
  `redop` and `root`, plus the common keys `node`, `namespace`, `test_name`, `topology=collective`,
  `nccl_version`, `nccl_library_version`, `nccl_headers_version`, `nccl_tests_version`,
  `cuda_version`, `image`, `job_id`, `uuid`, `started_at`, `ended_at` and `tag_*`.
  `namespace` is the pod namespace when the reporter can discover it, from the service account
  file or the first `resolv.conf` search domain, and is absent otherwise.
- `nccl-test-config` has the listed environment variables below with `source=environment`, the test's settings
  with `source=test_config`, `argv` with `source=command_line`, `error_message` with
  `source=errors` with up to 4000 characters each, and system and library versions with `source=library`. The library rows are
  `kernel_release`, `os_release`, `driver_version`, `cuda_driver_version`, `mpi_library_version`,
  `ucx_version` and `hpcx_version` where available, `pkg_<name>` for the installed verbs, provider,
  rdma-core, NCCL and DOCA-OFED packages, plus one row per loaded verbs, NCCL, NCCL network plugin,
  UCX, MPI, SHARP, HCOLL, CUDA and NVML library, named after the library and valued with its
  resolved file name, for example `libibverbs` with `libibverbs.so.1.14.56.0`. The config rows describe the
  reporting rank's node, and the node-facts rows every node. Libraries that are not already loaded are never loaded for this.
- `nccl-test-node-facts` has one `gpu` row per GPU. The value is the device UUID, else the
  serial, else the PCI bus id. Context carries `node`, `gpu`, `uuid`, `serial`, `pci_bus_id`,
  `device_name`, `rank`, `group`, `vbios`, `gpu_numa_node`, and the `cpu_affinity` and
  `cpu_numa_nodes` of the process that drives the GPU. Each node also gets the library rows
  described above, with `source=library`, from the first process seen on that node. The UUID is the identity because some boards report one
  NVML serial for the GPUs of a module.
