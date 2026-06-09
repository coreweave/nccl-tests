#!/usr/bin/env python3
"""Run distributed NCCL tests via mpirun from an MPIJob launcher pod.

Invoke manually after workers are ready, e.g.:

    python3 /opt/nccl-tests/run_test.py --profile h100-ib
    python3 /opt/nccl-tests/run_test.py --profile h100-ib --dry-run

MPI process count (-np) defaults to WORKERS * GPUS from the launcher pod env.
NCCL / UCX tuning vars use profile defaults unless the same-named env var is set
on the launcher pod (e.g. NCCL_SOCKET_IFNAME=eth1 overrides the default eth0).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Literal, Mapping

NCCL_TEST_BIN = "/opt/nccl_tests/build/all_reduce_perf"
DEFAULT_BENCHMARK_ARGS = ["-b", "512M", "-e", "8G", "-f", "2", "-g", "1"]
SPECTRUM_X_LD_PATH = "/opt/hpcx/nccl_spectrum-x_plugin/lib"


def _ucx_ib_devices(count: int) -> str:
    return ",".join(f"ibp{i}:1" for i in range(count))


def _spectrum_x_ld_library_path() -> str:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        return f"{SPECTRUM_X_LD_PATH}:{existing}"
    return SPECTRUM_X_LD_PATH


@dataclass(frozen=True)
class Profile:
    """mpirun configuration for a hardware / network topology."""

    mpi_np: int | None = 64
    bind_to_none: bool = True
    ld_library_path: Literal["forward", "spectrum_x"] = "forward"
    env_blocks: tuple[Mapping[str, str], ...] = ()
    env_extra: Mapping[str, str] = field(default_factory=dict)


def _env_var(name: str, default: str) -> str:
    """Use launcher env when set, otherwise the profile default."""
    return os.environ.get(name, default)


def _env_block(defaults: Mapping[str, str]) -> dict[str, str | None]:
    return {name: _env_var(name, value) for name, value in defaults.items()}


def _materialize_env(profile: Profile) -> dict[str, str | None]:
    merged: dict[str, str | None] = {}
    for block in profile.env_blocks:
        merged.update(_env_block(block))
    merged.update(_env_block(profile.env_extra))
    return merged


# Shared NCCL / UCX defaults; each key is overridable via the same-named env var.
_ENV_IB_BASE_DEFAULTS: dict[str, str] = {
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_IB_HCA": "ibp",
}
_ENV_NCCL_PLUGIN_NONE_DEFAULTS: dict[str, str] = {
    "NCCL_NET_PLUGIN": "none",
}
_ENV_UCX_IB_H100_DEFAULTS: dict[str, str] = {
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
_ENV_GB200_MNNVL_DEFAULTS: dict[str, str] = {
    "NVIDIA_IMEX_CHANNELS": "0",
    "NCCL_NET_GDR_C2C": "1",
    "NCCL_MNNVL_ENABLE": "1",
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_SHM_DISABLE": "0",
}
_ENV_GB300_ROCE_DEFAULTS: dict[str, str] = {
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

PROFILES: dict[str, Profile] = {
    "h100-ib": Profile(
        env_blocks=(
            _ENV_IB_BASE_DEFAULTS,
            _ENV_UCX_IB_H100_DEFAULTS,
            _ENV_SHARP_PCI_ORDERING_DEFAULTS,
            _ENV_NCCL_PLUGIN_NONE_DEFAULTS,
        ),
        env_extra={"NCCL_COLLNET_ENABLE": "0"},
    ),
    "h100-sharp": Profile(
        env_blocks=(
            _ENV_IB_BASE_DEFAULTS,
            _ENV_UCX_IB_H100_DEFAULTS,
            _ENV_SHARP_PCI_ORDERING_DEFAULTS,
        ),
        env_extra={
            "NCCL_ALGO": "COLLNETCHAIN",
            "NCCL_COLLNET_ENABLE": "1",
        },
    ),
    "gb200-nvl": Profile(
        mpi_np=None,
        env_blocks=(
            _ENV_IB_BASE_DEFAULTS,
            _ENV_GB200_MNNVL_DEFAULTS,
            _ENV_UCX_TCP_DEFAULTS,
            _ENV_NCCL_PLUGIN_NONE_DEFAULTS,
        ),
    ),
    "gb300-roce": Profile(
        mpi_np=None,
        ld_library_path="spectrum_x",
        env_blocks=(
            _ENV_IB_BASE_DEFAULTS,
            _ENV_GB300_ROCE_DEFAULTS,
            _ENV_UCX_TCP_DEFAULTS,
            _ENV_NCCL_PLUGIN_NONE_DEFAULTS,
        ),
    ),
}

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
    if "LD_LIBRARY_PATH" in os.environ:
        resolved["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    elif profile.ld_library_path == "spectrum_x":
        resolved["LD_LIBRARY_PATH"] = _spectrum_x_ld_library_path()
    else:
        resolved["LD_LIBRARY_PATH"] = None
    return resolved


def _export_args(env: Mapping[str, str | None]) -> list[str]:
    args: list[str] = []
    for key, value in env.items():
        if value is None:
            args.extend(["-x", key])
        else:
            args.extend(["-x", f"{key}={value}"])
    return args


def build_mpirun_command(
    profile_name: str,
    *,
    mpi_np: int | None = None,
    benchmark_args: list[str] | None = None,
) -> list[str]:
    if profile_name not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown profile {profile_name!r}; choose one of: {known}")

    profile = PROFILES[profile_name]
    resolved_np = _resolve_mpi_np(profile, mpi_np)
    test_args = benchmark_args if benchmark_args is not None else list(DEFAULT_BENCHMARK_ARGS)

    cmd = ["mpirun"]
    if resolved_np is not None:
        cmd.extend(["-np", str(resolved_np)])
    if profile.bind_to_none:
        cmd.extend(["-bind-to", "none"])
    cmd.extend(_export_args(_resolve_env(profile)))
    cmd.append(NCCL_TEST_BIN)
    cmd.extend(test_args)
    return cmd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the mpirun command and exit without running",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="list available profiles and exit",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        help="hardware/network profile (required unless --list-profiles)",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=None,
        help="MPI process count (default: NP env, WORKERS*GPUS, or profile default)",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help=f"override NCCL test args (default: {' '.join(DEFAULT_BENCHMARK_ARGS)})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_profiles:
        for name in sorted(PROFILES):
            print(name)
        return 0

    if not args.profile:
        print("error: --profile is required", file=sys.stderr)
        return 2

    benchmark_args = args.benchmark_args or None
    if benchmark_args is not None and benchmark_args and benchmark_args[0] == "--":
        benchmark_args = benchmark_args[1:]

    cmd = build_mpirun_command(
        args.profile,
        mpi_np=args.np,
        benchmark_args=benchmark_args,
    )
    print(f"profile: {args.profile}")
    print(f"command: {shlex.join(cmd)}")

    if args.dry_run:
        return 0

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
