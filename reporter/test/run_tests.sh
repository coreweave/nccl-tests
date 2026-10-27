#!/usr/bin/env bash
# GPU-free tests for the reporter. A fake nccl-tests binary, test/fake_run.cc, drives the
# hooks exactly as all_reduce_perf would, and a mock endpoint validates every row against
# the columnar results API rules. Run via `make test`.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/fake_run
[[ -x $BIN ]] || { echo "build first: make build/fake_run"; exit 2; }

WORK=$(mktemp -d)
MOCK=0
trap '[[ $MOCK -gt 0 ]] && kill $MOCK 2>/dev/null; rm -rf "$WORK"' EXIT
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')
python3 test/mock_endpoint.py "$PORT" "$WORK/m" &
MOCK=$!
for _ in $(seq 50); do
  python3 -c "import socket;socket.create_connection(('127.0.0.1',$PORT),0.2)" 2>/dev/null && break
  sleep 0.1
done
URL="http://127.0.0.1:$PORT/v2/runs/{run_id}/results"

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "ok    $*"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL  $*"; }
check() { local d=$1; shift; if eval "$@"; then ok "$d"; else fail "$d"; fi; }

# run NAME [VAR=value ...] runs the fake with a clean reporter environment plus the given variables.
RC=0; ELAPSED=0
run() {
  local name=$1; shift
  local t0; t0=$(date +%s.%N)
  # Scrub ambient reporter settings, every variable under a recorded prefix and the Slurm job
  # id so row counts do not depend on the surrounding environment. Container images set NCCL_*.
  local -a scrub=(); local v
  for v in $(env | grep -oE '^((NCCL|UCX|UCC|HCOLL|SHARP|OMPI|OPAL|PRTE|PMIX|PMI|MPICH|I_MPI|CUDA|GDRCOPY|MLX5|RDMAV|IBV)_[A-Za-z0-9_]*|SLURM_JOB_ID)=' | tr -d '='); do scrub+=(-u "$v"); done
  env "${scrub[@]}" NCCL_TESTS_REPORT_URL_FILE=/nonexistent NCCL_TESTS_REPORT_NAMESPACE_FILE=/nonexistent "$@" \
      "$BIN" all_reduce_perf -b 8 -e 1G -g 4 ${EXTRA_ARGS:-} >"$WORK/$name.out" 2>"$WORK/$name.err"
  RC=$?
  ELAPSED=$(python3 -c "import time; print(time.time()-$t0)")
}
npaths() { if [[ -f $WORK/m.paths ]]; then wc -l < "$WORK/m.paths"; else echo 0; fi; }
validation() { cat "$WORK/m.validation" 2>/dev/null || echo "NONE"; }
rows() { python3 test/rows.py "$WORK/m.body" "$@"; }
reset_mock() { rm -f "$WORK"/m.*; }
nartifacts() { if [[ -f $WORK/m.artifacts ]]; then wc -l < "$WORK/m.artifacts"; else echo 0; fi; }
results_run_id() { sed -n 's|^/v2/runs/\([^/]*\)/results$|\1|p' "$WORK/m.paths" 2>/dev/null | head -1; }

echo "# inactive: no endpoint provisioned"
run t1
check "inactive: exit code passes through"      '[[ $RC -eq 0 ]]'
check "inactive: no request made"               '[[ $(npaths) -eq 0 ]]'
check "inactive: stderr empty"                  '[[ ! -s $WORK/t1.err ]]'
check "inactive: stdout is the test banner"     'grep -q "Collective test concluded" $WORK/t1.out'

echo "# active: delivery and row content"
reset_mock
run t2 NCCL_TESTS_REPORT_URL="$URL" NCCL_DEBUG=INFO NCCL_IB_HCA=mlx5_0 NCCL_VERSION=2.30.4-1 \
       NCCL_COMM_ID=10.0.0.1:1234 NCCL_TOPO_FILE=/home/user/topo.xml NCCL_MY_SECRET=x \
       NCCL_API_TOKEN=t NCCL_IB_PKEY=0x8001 UCX_TLS=rc_x,cuda_copy UCX_IB_PKEY=0x8002 \
       OMPI_MCA_pml=ucx OMPI_MCA_extra="Bearer abc" SLURM_JOB_USER=alice MY_VAR=1 \
       NCCL_TESTS_REPORT_TAG_SKU=gb300 NCCL_TESTS_REPORT_IMAGE=img:1 NCCL_TESTS_REPORT_NAMESPACE=team-a SLURM_JOB_ID=42
check "active: exactly one request"             '[[ $(npaths) -eq 1 ]]'
check "active: rows pass the API rules"         '[[ $(validation) == PASS ]]'
check "active: run id is a UUIDv7 in the path"  'grep -Eq "^/v2/runs/[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/results$" $WORK/m.paths'
check "active: stdout identical to inactive"    'cmp -s $WORK/t1.out $WORK/t2.out'
check "active: stderr empty"                    '[[ ! -s $WORK/t2.err ]]'
check "metric rows: 10 sizes x 7 + 4 run rows"  '[[ $(rows count nccl-test-metric) -eq 74 ]]'
check "in-place: bus and algorithm bandwidth and latency" '[[ $(rows count nccl-test-metric bandwidth_in_place) -eq 10 && $(rows count nccl-test-metric alg_bandwidth_in_place) -eq 10 && $(rows count nccl-test-metric latency_in_place) -eq 10 ]]'
check "out-of-place: algorithm bandwidth"        '[[ $(rows count nccl-test-metric alg_bandwidth) -eq 10 ]]'
check "out_of_bounds row from the footer"        '[[ $(rows value nccl-test-metric out_of_bounds) == 0.0 || $(rows value nccl-test-metric out_of_bounds) == 0 ]]'
check "no min_bandwidth_check row without -a"    '[[ $(rows count nccl-test-metric min_bandwidth_check) -eq 0 ]]'
check "no cycle key on a single cycle"           '[[ "$(rows ctx nccl-test-metric latency cycle message_size=4K)" == None ]]'
check "config rows: 10 recorded variables + 12 settings + argv" '[[ $(rows count_except_source nccl-test-config library) -eq 23 ]]'
check "library: kernel and OS release recorded"  '[[ -n "$(rows value nccl-test-config kernel_release)" && $(rows ctx nccl-test-config kernel_release source) == library ]]'
check "library: OS release recorded"             '[[ -n "$(rows value nccl-test-config os_release)" ]]'
check "library: at least one library row" '[[ $(rows count_source nccl-test-config library) -ge 1 ]]'
check "facts: vbios, gpu numa node and cpu placement per GPU" '[[ $(rows ctx nccl-test-node-facts gpu vbios gpu=0) == 97.00.D0.00.5B && $(rows ctx nccl-test-node-facts gpu cpu_affinity gpu=2) == 72-143 && $(rows ctx nccl-test-node-facts gpu cpu_numa_nodes gpu=2) == 1 ]]'
check "facts: gpu numa node recorded"            '[[ "$(rows ctx nccl-test-node-facts gpu gpu_numa_node gpu=3)" == 1 || "$(rows ctx nccl-test-node-facts gpu gpu_numa_node gpu=3)" == 1.0 ]]'
check "facts: unknown vbios left out"            '[[ "$(rows ctx nccl-test-node-facts gpu vbios gpu=1)" == None ]]'
check "facts: library versions for each node"    '[[ $(rows count_source nccl-test-node-facts library) -ge 2 && -n "$(rows value nccl-test-node-facts kernel_release node=fakehost-02)" ]]'
check "facts rows: one per GPU"                 '[[ $(rows count nccl-test-node-facts gpu) -eq 4 ]]'
check "mean bandwidth row carries the footer"   '[[ $(rows value nccl-test-metric bandwidth agg=mean) == 150.25 ]]'
check "verdict PASS on a clean run"             '[[ $(rows ctx nccl-test-metric bandwidth verdict agg=mean) == PASS ]]'
check "returncode row is 0"                     '[[ $(rows value nccl-test-metric returncode) == 0 ]]'
check "per-size bandwidth is the oop bus_bw"    '[[ $(rows value nccl-test-metric bandwidth agg=sample message_size=8) == 100 ]]'
check "message_size humanized (4K)"             '[[ $(rows value nccl-test-metric latency message_size=4K) == 21.5 ]]'
check "tag_* from NCCL_TESTS_REPORT_TAG_*"      '[[ $(rows ctx nccl-test-metric returncode tag_sku) == gb300 ]]'
check "image, job_id carried"                   '[[ $(rows ctx nccl-test-metric returncode image) == img:1 && $(rows ctx nccl-test-metric returncode job_id) == 42 ]]'
check "namespace carried on the configured-URL path" '[[ $(rows ctx nccl-test-metric returncode namespace) == team-a ]]'
check "nccl_version is the loaded library, not NCCL_VERSION" '[[ $(rows ctx nccl-test-metric returncode nccl_version) == 2.31.2 && $(rows ctx nccl-test-metric returncode nccl_library_version) == 2.31.2 && $(rows value nccl-test-config NCCL_VERSION) == 2.30.4-1 ]]'
check "headers and cuda versions dotted"        '[[ $(rows ctx nccl-test-metric returncode nccl_headers_version) == 2.21.4 && $(rows ctx nccl-test-metric returncode cuda_version) == 12.4 ]]'
check "test_name from the program name"         '[[ $(rows ctx nccl-test-metric returncode test_name) == all_reduce_perf ]]'
check "run uuid from the footer"                '[[ $(rows ctx nccl-test-metric returncode uuid) == 11111111-2222-4333-8444-555555555555 ]]'
check "config: NCCL_DEBUG from environment"     '[[ $(rows value nccl-test-config NCCL_DEBUG) == INFO && $(rows ctx nccl-test-config NCCL_DEBUG source) == environment ]]'
check "config: reporter's own variables excluded" '! rows names nccl-test-config | grep -q NCCL_TESTS_REPORT'
check "config: NCCL_IB_HCA recorded"            '[[ $(rows value nccl-test-config NCCL_IB_HCA) == mlx5_0 ]]'
check "config: every set NCCL_* recorded"       '[[ $(rows value nccl-test-config NCCL_COMM_ID) == 10.0.0.1:1234 && $(rows value nccl-test-config NCCL_TOPO_FILE) == /home/user/topo.xml ]]'
check "config: UCX and MPI settings recorded"   '[[ $(rows value nccl-test-config UCX_TLS) == rc_x,cuda_copy && $(rows value nccl-test-config OMPI_MCA_pml) == ucx ]]'
check "config: credential-shaped names dropped" '! rows names nccl-test-config | grep -q -E "NCCL_MY_SECRET|NCCL_API_TOKEN"'
check "config: partition keys kept despite KEY" '[[ $(rows value nccl-test-config NCCL_IB_PKEY) == 0x8001 && $(rows value nccl-test-config UCX_IB_PKEY) == 0x8002 ]]'
check "config: credential-shaped values redacted" '[[ "$(rows value nccl-test-config OMPI_MCA_extra)" == "<redacted>" ]]'
check "config: other prefixes not recorded"     '! rows names nccl-test-config | grep -q -E "SLURM_JOB_USER|MY_VAR"'
check "config: step_factor, not step_bytes"     '[[ $(rows value nccl-test-config step_factor) == 2 ]] && ! rows names nccl-test-config | grep -q step_bytes'
check "config: argv recorded"                   '[[ "$(rows value nccl-test-config argv)" == *"all_reduce_perf -b 8 -e 1G -g 4"* ]]'
check "facts: serial when no uuid"              '[[ $(rows value nccl-test-node-facts gpu gpu=1) == 1650123456789 ]]'
check "facts: PCI bus id when neither"          '[[ $(rows value nccl-test-node-facts gpu gpu=2) == 0000:1d:00 ]]'
check "facts: uuid when known"                  '[[ $(rows value nccl-test-node-facts gpu gpu=0) == GPU-663051f8-43a9-e63c-6886-da4b882cd700 ]]'
check "facts: uuid wins over a shared serial"   '[[ $(rows value nccl-test-node-facts gpu gpu=3) == GPU-663051f8-43a9-e63c-6886-da4b882cd703 && $(rows ctx nccl-test-node-facts gpu serial gpu=3) == 1650123456789 ]]'
check "facts: no uuid context when unknown"     '[[ $(rows ctx nccl-test-node-facts gpu uuid gpu=1) == None ]]'
check "facts: device node wins, name trimmed"   '[[ $(rows ctx nccl-test-node-facts gpu node gpu=0) == fakehost-01 && "$(rows ctx nccl-test-node-facts gpu device_name gpu=0)" == "NVIDIA GB300" ]]'

echo "# only the writer rank sends"
reset_mock
run t3 NCCL_TESTS_REPORT_URL="$URL" FAKE_WRITER=0
check "non-writer: no request"                  '[[ $(npaths) -eq 0 && $RC -eq 0 ]]'

echo "# unreachable endpoint"
run t4 NCCL_TESTS_REPORT_URL="http://127.0.0.1:9/x" NCCL_TESTS_REPORT_CONNECT_TIMEOUT=1 NCCL_TESTS_REPORT_MAX_TIME=2
check "unreachable: exit code unchanged"        '[[ $RC -eq 0 ]]'
check "unreachable: silent"                     '[[ ! -s $WORK/t4.err ]]'
check "unreachable: bounded delay (<3 s)"       'python3 -c "import sys; sys.exit(0 if $ELAPSED < 3 else 1)"'

echo "# failing test"
reset_mock
run t5 NCCL_TESTS_REPORT_URL="$URL" FAKE_RC=7 FAKE_ERR="NCCL error: unhandled system error"
check "failing: exit code 7 passes through"     '[[ $RC -eq 7 ]]'
check "failing: verdict FAIL, returncode 7"     '[[ $(rows ctx nccl-test-metric returncode verdict) == FAIL && $(rows value nccl-test-metric returncode) == 7 ]]'
check "failing: error_message config row"       '[[ $(rows count nccl-test-config error_message) -eq 1 ]]'

echo "# wrong elements"
reset_mock
run t6 NCCL_TESTS_REPORT_URL="$URL" FAKE_WRONG=1
check "wrong_elements summed per size"          '[[ $(rows value nccl-test-metric wrong_elements message_size=64) == 3 ]]'

echo "# client-side limits"
reset_mock
BIG=$(head -c 40000 /dev/zero | tr '\0' a)
TAGS=(); for i in $(seq 70); do TAGS+=("NCCL_TESTS_REPORT_TAG_T$(printf %03d "$i")=v"); done
run t7 NCCL_TESTS_REPORT_URL="$URL" "NCCL_TESTS_REPORT_TAG_BIG=$BIG" "NCCL_TESTS_REPORT_TAG_Weird-Name=x" "NCCL_IB_HCA=$BIG$BIG" "${TAGS[@]}"
check "limits: fake ran and one request arrived" '[[ $RC -eq 0 && $(npaths) -eq 1 ]]'
check "limits: rows still pass the API rules"   '[[ $(validation) == PASS ]]'
check "limits: strings clipped to 32 KiB"       'x=$(rows maxstr) && [[ $x -le 32768 ]]'
check "limits: at most 64 context keys"         'x=$(rows maxkeys) && [[ $x -le 64 ]]'

echo "# switches"
reset_mock
run t8 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT=0
check "NCCL_TESTS_REPORT=0 disables"            '[[ $(npaths) -eq 0 ]]'
run t9 NCCL_TESTS_REPORT_URL=""
check "empty URL disables"                      '[[ $(npaths) -eq 0 ]]'
echo "$URL" > "$WORK/url"
run t10 NCCL_TESTS_REPORT_URL_FILE="$WORK/url"
check "URL from a mounted file"                 '[[ $(npaths) -eq 1 ]]'
reset_mock
run t11 NCCL_TESTS_REPORT_URL="$URL"; run t12 NCCL_TESTS_REPORT_URL="$URL"
check "two runs get two distinct run ids"       '[[ $(npaths) -eq 2 && $(sort -u $WORK/m.paths | wc -l) -eq 2 ]]'
reset_mock
run t13 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_RUN_ID=my-run-1
check "run id override in the path"             '[[ $(cat $WORK/m.paths) == /v2/runs/my-run-1/results ]]'
reset_mock
run t14 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_DEBUG=1 NCCL_TESTS_REPORT_AUTH_HEADER="X-Test: secret"
check "debug: logs to stderr"                   'grep -q "\[nccl-tests-ext\] delivered" $WORK/t14.err'
check "debug: stdout still identical"           'cmp -s $WORK/t1.out $WORK/t14.out'
check "auth header forwarded"                   'grep -q "auth=secret" $WORK/m.meta'

echo "# default-on: detection by resolving a collector host, only inside a cluster"
printf 'nameserver 10.0.0.10
search team-a.svc.cluster.local svc.cluster.local cluster.local
' > "$WORK/resolv.k8s"
printf 'nameserver 192.168.1.1
search home.lan
' > "$WORK/resolv.laptop"
printf 'nameserver 10.0.0.10
search verify-b.svc.cluster.local svc.cluster.local cluster.local
' > "$WORK/resolv.skip"
reset_mock
run d1 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.k8s" NCCL_TESTS_REPORT_HOSTS="127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT"
check "default-on: in a cluster, host resolves, request sent"   '[[ $RC -eq 0 && $(npaths) -eq 1 ]]'
check "default-on: default path and run id used"                'grep -Eq "^/v2/runs/[0-9a-f-]{36}/results$" $WORK/m.paths'
check "default-on: rows pass the API rules"                      '[[ $(validation) == PASS ]]'
check "default-on: stdout still identical"                       'cmp -s $WORK/t1.out $WORK/d1.out'
reset_mock
run d2 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.k8s" NCCL_TESTS_REPORT_HOSTS="no-such-host.invalid,127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT"
check "default-on: first candidate fails, second resolves"       '[[ $(npaths) -eq 1 ]]'
reset_mock
run d3 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.laptop" NCCL_TESTS_REPORT_HOSTS="127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT"
check "default-on: outside a cluster, no request"                '[[ $(npaths) -eq 0 && $RC -eq 0 ]]'
run d4 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.k8s" NCCL_TESTS_REPORT_HOSTS="no-such-host.invalid" NCCL_TESTS_REPORT_PORT="$PORT"
check "default-on: no candidate resolves, no request, exit unchanged" '[[ $(npaths) -eq 0 && $RC -eq 0 ]]'
check "default-on: unresolved lookup stays quick (<5 s)"         'python3 -c "import sys; sys.exit(0 if $ELAPSED < 5 else 1)"'
run d5 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.k8s"
check "default-on: no candidate hosts configured, no request"    '[[ $(npaths) -eq 0 ]]'
echo "# skip list"
run d6 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.k8s" NCCL_TESTS_REPORT_HOSTS="127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT" NCCL_TESTS_REPORT_NAMESPACE="verify-a" NCCL_TESTS_REPORT_SKIP_NAMESPACES="verify-a,verify-b"
check "skip list: namespace from env is skipped"                 '[[ $(npaths) -eq 0 ]]'
run d7 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.skip" NCCL_TESTS_REPORT_HOSTS="127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT" NCCL_TESTS_REPORT_SKIP_NAMESPACES="verify-a,verify-b"
check "skip list: namespace from resolv.conf search domain is skipped" '[[ $(npaths) -eq 0 ]]'
run d8 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.skip" NCCL_TESTS_REPORT_HOSTS="127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT" NCCL_TESTS_REPORT_SKIP_NAMESPACES="verify-a,verify-b" NCCL_TESTS_REPORT=1
check "skip list: NCCL_TESTS_REPORT=1 overrides the skip"        '[[ $(npaths) -eq 1 ]]'
reset_mock
run d9 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.laptop" NCCL_TESTS_REPORT_HOSTS="127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT" NCCL_TESTS_REPORT=1
check "forced: NCCL_TESTS_REPORT=1 reports outside a cluster"    '[[ $(npaths) -eq 1 ]]'
run d10 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.k8s" NCCL_TESTS_REPORT_HOSTS="127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT" NCCL_TESTS_REPORT_DEBUG=1
check "debug: reason and resolved endpoint logged"               'grep -q "in a cluster, endpoint resolved at finalize" $WORK/d10.err && grep -q "endpoint http://127.0.0.1:$PORT/v2/runs/" $WORK/d10.err'
run d11 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.laptop" NCCL_TESTS_REPORT_DEBUG=1
check "debug: inactive reason logged"                            'grep -q "inactive, no endpoint configured and no candidate hosts" $WORK/d11.err'

echo "# review fixes"
reset_mock
run e1 NCCL_TESTS_REPORT_URL="" NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.k8s" NCCL_TESTS_REPORT_HOSTS="127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT"
check "opt-out: empty NCCL_TESTS_REPORT_URL beats candidate hosts" '[[ $(npaths) -eq 0 && $RC -eq 0 ]]'
reset_mock
run e2 NCCL_TESTS_REPORT_URL="$URL" FAKE_NSIZES=5000
check "cap: more than 4096 result lines sends nothing, exit unchanged" '[[ $(npaths) -eq 0 && $RC -eq 0 ]]'
reset_mock
run e3 NCCL_TESTS_REPORT_URL="$URL" FAKE_CPUTIME=1
check "cpu time: no latency rows under -C 1"     '[[ $(npaths) -eq 1 && $(rows count nccl-test-metric latency) -eq 0 && $(rows count nccl-test-metric latency_in_place) -eq 0 ]]'
check "cpu time: sent as cpu_time instead"       '[[ $(rows count nccl-test-metric cpu_time) -eq 10 && $(rows count nccl-test-metric cpu_time_in_place) -eq 10 ]]'
check "cpu time: bandwidth rows still present"   '[[ $(rows count nccl-test-metric bandwidth) -eq 11 ]]'
reset_mock
run e5 NCCL_TESTS_REPORT_URL="$URL" FAKE_CYCLES=3 FAKE_CHECK=120.5 FAKE_OOB=2
check "cycles: run-level cycles row"             '[[ $(rows value nccl-test-metric cycles) == 3.0 || $(rows value nccl-test-metric cycles) == 3 ]]'
check "cycles: rows tagged with their cycle"     '[[ "$(rows ctx nccl-test-metric latency cycle message_size=4K cycle=2)" == 2 || "$(rows ctx nccl-test-metric latency cycle message_size=4K cycle=2.0)" == 2.0 ]]'
check "footer: min_bandwidth_check and out_of_bounds" '[[ $(rows value nccl-test-metric min_bandwidth_check) == 120.5 && ( $(rows value nccl-test-metric out_of_bounds) == 2 || $(rows value nccl-test-metric out_of_bounds) == 2.0 ) ]]'
reset_mock
run e6 NCCL_TESTS_REPORT_URL="$URL" FAKE_TYPES=2
check "types: a sweep per type is not a cycle"   '[[ $(rows count nccl-test-metric cycles) -eq 0 && "$(rows ctx nccl-test-metric latency cycle message_size=4K dtype=half)" == None ]]'
reset_mock
run e7 NCCL_TESTS_REPORT_URL="$URL" FAKE_TYPES=2 FAKE_CYCLES=2
check "types: cycles counted within each type"   '[[ ( $(rows value nccl-test-metric cycles) == 2 || $(rows value nccl-test-metric cycles) == 2.0 ) && "$(rows ctx nccl-test-metric latency cycle message_size=4K dtype=half cycle=1)" == 1* ]]'
reset_mock
run e4 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_MAX_TIME=0 NCCL_TESTS_REPORT_CONNECT_TIMEOUT=0
check "timeouts: zero falls back to the defaults and still sends" '[[ $(npaths) -eq 1 ]]'

echo "# log upload"
AURL="http://127.0.0.1:$PORT/v1/runs/{run_id}/artifacts/{file}"
SHORT=$(hostname | cut -d. -f1)
printf 'NCCL INFO fake debug line\n' > "$WORK/dbg.$SHORT.log"
printf '{"fake": true}\n' > "$WORK/out.json"
reset_mock
run l1 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_ARTIFACT_URL="$AURL" NCCL_DEBUG_FILE="$WORK/dbg.%h.log"
check "logs: off unless NCCL_TESTS_REPORT_LOGS=1"   '[[ $(nartifacts) -eq 0 && $(npaths) -eq 1 && $(rows count nccl-test-config artifact) -eq 0 ]]'
reset_mock
EXTRA_ARGS="-J $WORK/out.json" run l2 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_LOGS=1 NCCL_TESTS_REPORT_ARTIFACT_URL="$AURL" NCCL_DEBUG_FILE="$WORK/dbg.%h.log"
check "logs: debug file and -J output both sent"    '[[ $RC -eq 0 && $(nartifacts) -eq 2 && $(npaths) -eq 1 ]]'
check "logs: artifacts share the results run id"    'id=$(results_run_id) && [[ -n $id && $(grep -c "^/v1/runs/$id/artifacts/" "$WORK/m.artifacts") -eq 2 ]]'
check "logs: debug file named by host and pid"      'grep -qE "/artifacts/nccl-debug-${SHORT//[^A-Za-z0-9_-]/_}-[0-9]+\.log$" "$WORK/m.artifacts"'
check "logs: debug file body is the file"           'f=$(ls "$WORK"/m.artifact.nccl-debug-*.log) && cmp -s "$f" "$WORK/dbg.$SHORT.log"'
check "logs: -J output sent as JSON"                'f=$(ls "$WORK"/m.artifact.nccl-tests-*.json) && cmp -s "$f" "$WORK/out.json" && [[ $(cat "$f.type") == application/json ]]'
check "logs: results list the artifacts"            '[[ $(rows count nccl-test-config artifact) -eq 2 && $(rows ctx nccl-test-config artifact source) == logs ]]'
check "logs: stdout unchanged"                      'cmp -s "$WORK/l1.out" "$WORK/l2.out"'
reset_mock
run l3 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_LOGS=1 NCCL_TESTS_REPORT_ARTIFACT_URL="$AURL" NCCL_DEBUG_FILE="$WORK/dbg.%h.log" FAKE_WRITER=0 FAKE_SHARED_RUN_ID=shared-run-1
check "logs: other ranks send under rank 0's run id" '[[ $(nartifacts) -eq 1 && $(npaths) -eq 0 ]] && grep -q "^/v1/runs/shared-run-1/artifacts/nccl-debug-" "$WORK/m.artifacts"'
reset_mock
run l4 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_LOGS=1 NCCL_TESTS_REPORT_ARTIFACT_URL="$AURL" NCCL_DEBUG_FILE="$WORK/dbg.%h.log" FAKE_WRITER=0 FAKE_NO_SHARE=1
check "logs: no shared run id, other ranks send nothing" '[[ $(nartifacts) -eq 0 && $RC -eq 0 ]]'
head -c 2097152 /dev/zero > "$WORK/big.log"
reset_mock
run l5 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_LOGS=1 NCCL_TESTS_REPORT_ARTIFACT_URL="$AURL" NCCL_DEBUG_FILE="$WORK/big.log" NCCL_TESTS_REPORT_LOG_MAX_MIB=1
check "logs: a file over the limit is skipped"      '[[ $(nartifacts) -eq 0 && $(npaths) -eq 1 && $(rows count nccl-test-config artifact) -eq 0 ]]'
reset_mock
run l6 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_LOGS=1 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.k8s" NCCL_TESTS_REPORT_ARTIFACT_HOSTS="127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT" NCCL_DEBUG_FILE="$WORK/dbg.%h.log"
check "logs: artifact host found in a cluster"      '[[ $(nartifacts) -eq 1 ]] && grep -q "^/v1/runs/.*/artifacts/nccl-debug-" "$WORK/m.artifacts"'
reset_mock
run l7 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_LOGS=1 NCCL_TESTS_REPORT_RESOLV_CONF="$WORK/resolv.laptop" NCCL_TESTS_REPORT_ARTIFACT_HOSTS="127.0.0.1" NCCL_TESTS_REPORT_PORT="$PORT" NCCL_DEBUG_FILE="$WORK/dbg.%h.log"
check "logs: no artifact endpoint outside a cluster" '[[ $(nartifacts) -eq 0 && $(npaths) -eq 1 && $RC -eq 0 ]]'
printf 'pct\n' > "$WORK/pct%.log"
reset_mock
run l8 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_LOGS=1 NCCL_TESTS_REPORT_ARTIFACT_URL="$AURL" NCCL_DEBUG_FILE="$WORK/pct%%.log"
check "logs: %% in NCCL_DEBUG_FILE is a literal %"  '[[ $(nartifacts) -eq 1 ]]'

LONG_ID=run-$(printf 'x%.0s' $(seq 70))
reset_mock
run l9 NCCL_TESTS_REPORT_URL="$URL" NCCL_TESTS_REPORT_RUN_ID="$LONG_ID"
check "run id: an override longer than the exchange stays whole" '[[ $(results_run_id) == "$LONG_ID" ]]'
FIXED_URL="http://127.0.0.1:$PORT/v2/runs/fixed-run/results"
reset_mock
run l10 NCCL_TESTS_REPORT_URL="$FIXED_URL" NCCL_TESTS_REPORT_LOGS=1 NCCL_TESTS_REPORT_ARTIFACT_URL="$AURL" NCCL_DEBUG_FILE="$WORK/dbg.%h.log"
check "logs: fixed results URL without a run id sends no logs" '[[ $(nartifacts) -eq 0 && $(npaths) -eq 1 ]]'
reset_mock
run l11 NCCL_TESTS_REPORT_URL="$FIXED_URL" NCCL_TESTS_REPORT_RUN_ID=fixed-run NCCL_TESTS_REPORT_LOGS=1 NCCL_TESTS_REPORT_ARTIFACT_URL="$AURL" NCCL_DEBUG_FILE="$WORK/dbg.%h.log"
check "logs: fixed results URL with the matching run id"        '[[ $(nartifacts) -eq 1 ]] && grep -q "^/v1/runs/fixed-run/artifacts/" "$WORK/m.artifacts"'

echo
echo "passed $PASS, failed $FAIL"
[[ $FAIL -eq 0 ]]
