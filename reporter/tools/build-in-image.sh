#!/usr/bin/env bash
# Compile the nccl-tests fork with and without the extension inside the official devel
# image. No GPU is needed to compile, only to run. Builds one GPU architecture by default,
# which NVCC_GENCODE overrides, so the check takes minutes, not an hour.
#
#   tools/build-in-image.sh [image] [fork-dir]
#
# Outputs go under build/image-build/ as build-<stock|ext>-mpi<0|1>.log and the all_reduce_perf
# binaries from each build.
set -euo pipefail
IMAGE=${1:-ghcr.io/coreweave/nccl-tests:13.2.1-devel-ubuntu24.04-nccl2.31.2-1-478d8e5}
FORK=${2:?usage: build-in-image.sh [image] <path to a nccl-tests-json checkout>}
EXT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$EXT/build/image-build
mkdir -p "$OUT"
echo "== pulling $IMAGE"
docker pull -q "$IMAGE"
docker run --rm -i \
  -v "$FORK":/src/fork:ro -v "$EXT":/src/ext:ro -v "$OUT":/out \
  -e GENCODE="${NVCC_GENCODE:--gencode=arch=compute_90,code=sm_90}" \
  "$IMAGE" bash -s <<'IN'
set -uo pipefail
fail=0
echo "== toolchain"; nvcc --version | tail -1; gcc --version | head -1
NCCL_H=$(find / -name nccl.h -path '*include*' 2>/dev/null | head -1)
MPI_H=$(find /opt /usr -name mpi.h -path '*include*' 2>/dev/null | head -1)
NCCL_HOME=$(dirname "$(dirname "$NCCL_H")"); MPI_HOME=$(dirname "$(dirname "$MPI_H")")
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
echo "NCCL_HOME=$NCCL_HOME MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME"
cp -r /src/fork /work-fork && cp -r /src/ext /work-ext && rm -rf /work-ext/build
echo "== extension library and tests, built with the image's compiler"
if make -s -C /work-ext lib && make -s -C /work-ext build/fake_run && (cd /work-ext && bash test/run_tests.sh | tail -1); then :; else echo "   library or tests FAILED"; fail=1; fi
for mode in stock ext; do for mpi in 0 1; do
  echo "== fork build: $mode MPI=$mpi"
  rm -rf /work-fork/build
  args=(MPI=$mpi NVCC_GENCODE="$GENCODE" NCCL_HOME="$NCCL_HOME" CUDA_HOME="$CUDA_HOME" VERBOSE=1)
  [[ $mpi == 1 ]] && args+=(MPI_HOME="$MPI_HOME" NAME_SUFFIX=_mpi)
  [[ $mode == ext ]] && args+=(EXT_DIR=/work-ext)
  log=/out/build-$mode-mpi$mpi.log
  if make -C /work-fork/src -j"$(nproc)" "${args[@]}" >"$log" 2>&1; then echo "   OK"; else echo "   FAILED"; grep -nE 'error|Error' "$log" | head -20; fail=1; fi
  echo "   warnings: $(grep -ci 'warning' "$log")"
  b=/work-fork/build/all_reduce_perf; [[ $mpi == 1 ]] && b=${b}_mpi
  if [[ -f $b ]]; then echo "   hook symbols in binary: $(nm -C "$b" 2>/dev/null | grep -c 'nccl_tests_ext_')"; cp "$b" "/out/all_reduce_perf-$mode-mpi$mpi"; else echo "   binary missing"; fail=1; fi
done; done
echo "== smoke, ext MPI=0 binary, no GPU present. Expect a CUDA error exit and no crash"
# A test failure exit is expected without a GPU. A missing binary or a signal, exit 126 or more,
# is a real failure. A binary that needs a driver library the image lacks is skipped.
smoke() {
  local b=$1; shift
  if [[ ! -x $b ]]; then echo "   $b missing"; fail=1; return; fi
  local missing; missing=$(ldd "$b" 2>/dev/null | awk '/not found/ {print $1}' | tr '\n' ' ')
  if [[ -n $missing ]]; then echo "   skipped, the image lacks ${missing% }, which needs a GPU driver"; return; fi
  "$@" "$b" -b 8 -e 16 -g 1; local rc=$?
  echo "   exit=$rc"
  if (( rc >= 126 )); then echo "   crashed"; fail=1; fi
}
smoke /out/all_reduce_perf-ext-mpi0 env NCCL_TESTS_REPORT_URL=http://127.0.0.1:9/x NCCL_TESTS_REPORT_DEBUG=1
echo "== smoke, stock MPI=0 binary, same"
smoke /out/all_reduce_perf-stock-mpi0 env
exit $fail
IN
