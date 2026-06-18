#!/usr/bin/env python3
"""Run distributed NCCL tests via mpirun from an MPIJob launcher pod.

Invoke manually after workers are ready, e.g.:

    python3 /opt/nccl-tests/run_test.py --platform hgx --fabric ib --collnet off

    With no ``-- …`` benchmark remainder, runs each template in ``BENCHMARK_COMMAND_TEMPLATES``
    (all_reduce, alltoall, reduce_scatter) in order. Pass ``-- <binary> [args…]`` to run a single benchmark.

    python3 /opt/nccl-tests/run_test.py --platform hgx --fabric ib --collnet off \\
        -- alltoall_perf -b 8M -e 64M -f 2 -g 1

    Reusable argv tails (binary + flags) live in ``BENCHMARK_TEMPLATE_ALLREDUCE``,
    ``BENCHMARK_TEMPLATE_ALLTOALL``, and ``BENCHMARK_TEMPLATE_REDUCE_SCATTER``.

MPI process count (-np) defaults to WORKERS * GPUS from the launcher pod env.

Environment variables for mpirun (-x) are assembled in layers (later wins):

  1. Stack defaults for known NCCL / UCX / SHARP / NVIDIA tuning keys
  2. Launcher pod env overrides stack keys when the same name is set
  3. Launcher pod env pass-through for new tuning keys (prefixes: NCCL_, UCX_,
     SHARP_, NVIDIA_, OMPI_MCA_) not already in the stack
  4. EXTRA_MPI_ENV launcher env (space/comma-separated KEY=VAL tokens)

Examples:

    # Launcher pod env in MPIJob YAML
    - name: NCCL_ALGO
      value: NVLSTREE

    - name: EXTRA_MPI_ENV
      value: "NCCL_DEBUG=INFO NCCL_ALGO=NVLSTREE"
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Literal, Mapping

NCCL_TESTS_BUILD_DIR = "/opt/nccl_tests/build"

# Full argv tails for `run_test.py … -- <argv>` or for pasting into MPIJob `mpirun` lines.
# With no `-- …` remainder, main runs every template in BENCHMARK_COMMAND_TEMPLATES in order.
BENCHMARK_TEMPLATE_ALLREDUCE: tuple[str, ...] = (
    os.path.join(NCCL_TESTS_BUILD_DIR, "all_reduce_perf"),
    "-b",
    "512M",
    "-e",
    "8G",
    "-f",
    "2",
    "-g",
    "1",
)
BENCHMARK_TEMPLATE_ALLTOALL: tuple[str, ...] = (
    os.path.join(NCCL_TESTS_BUILD_DIR, "alltoall_perf"),
    "-b",
    "8M",
    "-e",
    "64M",
    "-f",
    "2",
    "-g",
    "1",
)
BENCHMARK_TEMPLATE_REDUCE_SCATTER: tuple[str, ...] = (
    os.path.join(NCCL_TESTS_BUILD_DIR, "reduce_scatter_perf"),
    "-b",
    "512M",
    "-e",
    "8G",
    "-f",
    "2",
    "-g",
    "1",
)

# Ordered list of full [binary, …args] templates; used to pick default flags per binary.
BENCHMARK_COMMAND_TEMPLATES: tuple[tuple[str, ...], ...] = (
    BENCHMARK_TEMPLATE_ALLREDUCE,
    BENCHMARK_TEMPLATE_ALLTOALL,
    BENCHMARK_TEMPLATE_REDUCE_SCATTER,
)

SPECTRUM_X_LD_PATH = "/opt/hpcx/nccl_spectrum-x_plugin/lib"

# Basename only (resolved under NCCL_TESTS_BUILD_DIR); paths use / ./ ~/ explicitly.
_BENCHMARK_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Launcher env vars forwarded to workers when not already set by the stack defaults.
_MPI_ENV_PREFIXES = ("NCCL_", "UCX_", "SHARP_", "NVIDIA_", "OMPI_MCA_")

# Launcher-only vars that must not be exported to MPI ranks.
_LAUNCHER_ENV_BLOCKLIST = frozenset(
    {
        "WORKERS",
        "GPUS",
        "NP",
        "PROFILE",
        "NAMESPACE",
        "THREADS",
        "HOSTFILE",
        "EXTRA_MPI_ENV",
        "OMPI_ALLOW_RUN_AS_ROOT",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
    }
)

# Server family for --platform (not per-SKU): Hopper/HGX, Grace Blackwell, or VR / passthrough.
PLATFORMS = frozenset({"hgx", "gb", "vr"})
FABRICS = frozenset({"ib", "roce", "nvl"})
COLLNET_CHOICES = ("off", "sharp")


def _ucx_ib_devices(count: int) -> str:
    return ",".join(f"ibp{i}:1" for i in range(count))


def _spectrum_x_ld_library_path() -> str:
    """Spectrum-X plugin first, then launcher LD_LIBRARY_PATH (matches MPIJob YAML expansion)."""
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{SPECTRUM_X_LD_PATH}:{existing}"


@dataclass(frozen=True)
class Profile:
    """mpirun configuration for a hardware / network topology."""

    mpi_np: int | None = 64
    bind_to_none: bool = True
    ld_library_path: Literal["forward", "spectrum_x"] = "forward"
    env_blocks: tuple[Mapping[str, str], ...] = ()
    env_extra: Mapping[str, str] = field(default_factory=dict)


def _env_var(name: str, default: str) -> str:
    """Use launcher env when set, otherwise the stack default."""
    return os.environ.get(name, default)


def _env_block(defaults: Mapping[str, str]) -> dict[str, str | None]:
    return {name: _env_var(name, value) for name, value in defaults.items()}


def _materialize_env(profile: Profile) -> dict[str, str | None]:
    merged: dict[str, str | None] = {}
    for block in profile.env_blocks:
        merged.update(_env_block(block))
    merged.update(_env_block(profile.env_extra))
    return merged


# --- --platform + --fabric: family-only NCCL defaults (each key overridable via launcher env) ---
_ENV_GB200_MNNVL_DEFAULTS: dict[str, str] = {
    "NVIDIA_IMEX_CHANNELS": "0",
    "NCCL_NET_GDR_C2C": "1",
    "NCCL_MNNVL_ENABLE": "1",
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_SHM_DISABLE": "0",
}


def _platform_env_blocks(platform: str, fabric: str) -> tuple[Mapping[str, str], ...]:
    """Extra env blocks for --platform, optionally refined by --fabric (e.g. GB + nvl → MNNVL)."""
    if platform == "gb" and fabric == "nvl":
        return (_ENV_GB200_MNNVL_DEFAULTS,)
    return ()

# --- --fabric: network path (IB, RoCE, NVLink rack / host TCP UCX) ---
_ENV_IB_BASE_DEFAULTS: dict[str, str] = {
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_IB_HCA": "ibp",
}
_ENV_UCX_IB_HOPPER_DEFAULTS: dict[str, str] = {
    "UCX_NET_DEVICES": _ucx_ib_devices(8),
}
_ENV_SHARP_PCI_ORDERING_DEFAULTS: dict[str, str] = {
    "SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING": "1",
}
_ENV_UCX_TCP_DEFAULTS: dict[str, str] = {
    "UCX_TLS": "tcp",
    "UCX_NET_DEVICES": "eth0",
    "OMPI_MCA_coll_hcoll_enable": "0",
}
_ENV_NCCL_PLUGIN_NONE_DEFAULTS: dict[str, str] = {
    "NCCL_NET_PLUGIN": "none",
}
_ENV_ROCE_DEFAULTS: dict[str, str] = {
    "NCCL_IB_NET_LATENCY": "50",
    "NCCL_IB_ADAPTIVE_ROUTING": "1",
    "NCCL_NVLS_NCHANNELS": "24",
    "NCCL_P2P_NET_CHUNKSIZE": "524288",
    "NCCL_NVLSTREE_MAX_CHUNKSIZE": "262144",
    "NCCL_NVLS_CHUNKSIZE": "262144",
    "NCCL_IB_ADDR_FAMILY": "AF_INET6",
    "NCCL_IB_ADDR_RANGE": "fd02::/16",
    "NCCL_IB_TC": "96",
}

_FABRIC_ENV_BLOCKS: dict[str, tuple[Mapping[str, str], ...]] = {
    "ib": (
        _ENV_IB_BASE_DEFAULTS,
        _ENV_UCX_IB_HOPPER_DEFAULTS,
        _ENV_SHARP_PCI_ORDERING_DEFAULTS,
    ),
    "roce": (
        _ENV_IB_BASE_DEFAULTS,
        _ENV_ROCE_DEFAULTS,
        _ENV_UCX_TCP_DEFAULTS,
        _ENV_NCCL_PLUGIN_NONE_DEFAULTS,
    ),
    "nvl": (
        _ENV_IB_BASE_DEFAULTS,
        _ENV_UCX_TCP_DEFAULTS,
        _ENV_NCCL_PLUGIN_NONE_DEFAULTS,
    ),
}

# --- --collnet: SHARP / NCCL collnet (only applies with --fabric ib on Hopper) ---


def _collnet_env_blocks(fabric: str, collnet: str) -> tuple[Mapping[str, str], ...]:
    if fabric != "ib":
        return ()
    if collnet == "off":
        return (_ENV_NCCL_PLUGIN_NONE_DEFAULTS,)
    return ()


def _collnet_env_extra(fabric: str, collnet: str) -> dict[str, str]:
    if fabric != "ib":
        return {}
    if collnet == "off":
        return {"NCCL_COLLNET_ENABLE": "0"}
    return {
        "NCCL_ALGO": "COLLNETCHAIN",
        "NCCL_COLLNET_ENABLE": "1",
    }


def _mpi_np_for_stack(platform: str) -> int | None:
    if platform == "gb":
        return None
    return 64


def _ld_library_path_for_stack(platform: str, fabric: str) -> Literal["forward", "spectrum_x"]:
    if platform == "gb" and fabric == "roce":
        return "spectrum_x"
    return "forward"


def _compose_profile(platform: str, fabric: str, collnet: str) -> Profile:
    """Merge env in order: platform defaults, fabric defaults, collnet defaults."""
    env_blocks = (
        *_platform_env_blocks(platform, fabric),
        *_FABRIC_ENV_BLOCKS[fabric],
        *_collnet_env_blocks(fabric, collnet),
    )
    return Profile(
        mpi_np=_mpi_np_for_stack(platform),
        ld_library_path=_ld_library_path_for_stack(platform, fabric),
        env_blocks=env_blocks,
        env_extra=_collnet_env_extra(fabric, collnet),
    )


def _format_stack(platform: str, fabric: str, collnet: str) -> str:
    return f"--platform {platform} --fabric {fabric} --collnet {collnet}"


def _iter_stack_combinations():
    """Yield all --platform × --fabric × --collnet tuples (for --list-stacks)."""
    return itertools.product(sorted(PLATFORMS), sorted(FABRICS), COLLNET_CHOICES)


def _parse_env_assignments(raw: str) -> dict[str, str]:
    """Parse KEY=VAL tokens separated by whitespace or commas."""
    assignments: dict[str, str] = {}
    for token in re.split(r"[\s,]+", raw.strip()):
        if not token or "=" not in token:
            continue
        key, _, value = token.partition("=")
        key = key.strip()
        if key:
            assignments[key] = value.strip()
    return assignments


def _launcher_pass_through_env(stack_env: Mapping[str, str | None]) -> dict[str, str]:
    """Export new tuning vars from launcher env (e.g. NCCL_ALGO for a new NCCL release)."""
    extra: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _LAUNCHER_ENV_BLOCKLIST or key in stack_env:
            continue
        if any(key.startswith(prefix) for prefix in _MPI_ENV_PREFIXES):
            extra[key] = value
    return extra


def _resolve_mpi_np(profile: Profile, override: int | None) -> int | None:
    if override is not None:
        return override
    np_env = os.getenv("NP")
    if np_env:
        return int(np_env)
    workers = os.getenv("WORKERS")
    gpus = os.getenv("GPUS")
    if workers and gpus:
        return int(workers) * int(gpus)
    return profile.mpi_np


def _resolve_env(profile: Profile) -> dict[str, str | None]:
    resolved = _materialize_env(profile)
    if profile.ld_library_path == "spectrum_x":
        resolved["LD_LIBRARY_PATH"] = _spectrum_x_ld_library_path()
    elif "LD_LIBRARY_PATH" in os.environ:
        resolved["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    else:
        resolved["LD_LIBRARY_PATH"] = None

    for key, value in _launcher_pass_through_env(resolved).items():
        resolved[key] = value

    extra_mpi_env = os.getenv("EXTRA_MPI_ENV", "")
    if extra_mpi_env:
        resolved.update(_parse_env_assignments(extra_mpi_env))

    return resolved


def _export_args(env: Mapping[str, str | None]) -> list[str]:
    args: list[str] = []
    for key, value in env.items():
        if value is None:
            args.extend(["-x", key])
        else:
            args.extend(["-x", f"{key}={value}"])
    return args


def _default_benchmark_argv_for_binary(binary: str) -> list[str]:
    """Default perf argv for `binary`, from BENCHMARK_COMMAND_TEMPLATES basename match."""
    want = os.path.basename(binary)
    for template in BENCHMARK_COMMAND_TEMPLATES:
        if os.path.basename(template[0]) == want:
            return list(template[1:])
    return list(BENCHMARK_TEMPLATE_ALLREDUCE[1:])


def _resolve_benchmark_command(parts: list[str] | None) -> list[str]:
    """Build [binary, ...argv]. Basenames are resolved under NCCL_TESTS_BUILD_DIR.

    If ``parts`` is None or empty, returns ``BENCHMARK_TEMPLATE_ALLREDUCE`` (single
    default when only one mpirun is built via ``benchmark_args``).

    If only the binary is given (no perf flags), append defaults from
    BENCHMARK_COMMAND_TEMPLATES when the basename matches; otherwise use
    all_reduce-style defaults (BENCHMARK_TEMPLATE_ALLREDUCE[1:]).
    """
    if not parts:
        return list(BENCHMARK_TEMPLATE_ALLREDUCE)
    first, *rest = parts
    if "/" in first or first.startswith((".", os.sep)) or first.startswith("~/"):
        binary = os.path.expanduser(first)
    else:
        if not _BENCHMARK_BASENAME_RE.match(first):
            raise ValueError(
                f"invalid benchmark binary name {first!r}; "
                "use a basename (e.g. alltoall_perf) or an explicit path"
            )
        binary = os.path.join(NCCL_TESTS_BUILD_DIR, first)
    if not rest:
        rest = _default_benchmark_argv_for_binary(binary)
    return [binary, *rest]


def build_mpirun_command(
    platform: str,
    fabric: str,
    collnet: str,
    *,
    mpi_np: int | None = None,
    benchmark_args: list[str] | None = None,
    benchmark_cmd: list[str] | None = None,
) -> list[str]:
    profile = _compose_profile(platform, fabric, collnet)
    resolved_np = _resolve_mpi_np(profile, mpi_np)
    if benchmark_cmd is not None:
        bench = list(benchmark_cmd)
    else:
        bench = _resolve_benchmark_command(benchmark_args)

    cmd = ["mpirun"]
    if resolved_np is not None:
        cmd.extend(["-np", str(resolved_np)])
    if profile.bind_to_none:
        cmd.extend(["-bind-to", "none"])
    cmd.extend(_export_args(_resolve_env(profile)))
    cmd.extend(bench)
    return cmd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the mpirun command and exit without running",
    )
    parser.add_argument(
        "--list-stacks",
        action="store_true",
        help="list every --platform × --fabric × --collnet combination and exit",
    )
    parser.add_argument(
        "--platform",
        choices=sorted(PLATFORMS),
        help="server family: hgx (Hopper / HGX), gb (Grace Blackwell NVL72), vr (virtual / passthrough — tune via pod env)",
    )
    parser.add_argument(
        "--fabric",
        choices=sorted(FABRICS),
        help="network path: ib (HGX IB + UCX), roce (GB Spectrum-X RoCE tuning), nvl (NVLink rack + TCP UCX)",
    )
    parser.add_argument(
        "--collnet",
        choices=COLLNET_CHOICES,
        default="off",
        help="SHARP / NCCL collnet (only adds env when --fabric ib)",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=None,
        help="MPI process count (default: NP env, WORKERS*GPUS, or stack default)",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help=(
            "benchmark command: optional '--' then "
            "[BUILD_DIR/]<name>_perf [args…] or a path to the binary; "
            "basename only is resolved under "
            f"{NCCL_TESTS_BUILD_DIR!r}. "
            "If this remainder is omitted (or only '--'), each template in "
            "BENCHMARK_COMMAND_TEMPLATES is run in order. "
            "If only the binary is given, default perf flags match "
            f"{', '.join(os.path.basename(t[0]) for t in BENCHMARK_COMMAND_TEMPLATES)} "
            "templates when the name matches; otherwise the all_reduce_perf "
            "default sweep is used."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_stacks:
        for p, f, c in _iter_stack_combinations():
            print(_format_stack(p, f, c))
        return 0

    if not args.platform or not args.fabric:
        print(
            "error: --platform and --fabric are required",
            file=sys.stderr,
        )
        return 2

    benchmark_args = args.benchmark_args or None
    if benchmark_args is not None:
        benchmark_args = [a for a in benchmark_args if a]
        if benchmark_args and benchmark_args[0] == "--":
            benchmark_args = benchmark_args[1:]

    try:
        if benchmark_args is None:
            runs: list[tuple[str, list[str]]] = [
                (
                    os.path.basename(template[0]),
                    build_mpirun_command(
                        args.platform,
                        args.fabric,
                        args.collnet,
                        mpi_np=args.np,
                        benchmark_cmd=list(template),
                    ),
                )
                for template in BENCHMARK_COMMAND_TEMPLATES
            ]
        else:
            resolved = _resolve_benchmark_command(benchmark_args)
            runs = [
                (
                    os.path.basename(resolved[0]),
                    build_mpirun_command(
                        args.platform,
                        args.fabric,
                        args.collnet,
                        mpi_np=args.np,
                        benchmark_args=benchmark_args,
                    ),
                )
            ]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"stack: {_format_stack(args.platform, args.fabric, args.collnet)}")
    total = len(runs)
    for i, (label, cmd) in enumerate(runs, start=1):
        print(f"benchmark {i}/{total} ({label}):")
        print(f"command: {shlex.join(cmd)}")

    if args.dry_run:
        return 0

    for _, cmd in runs:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
