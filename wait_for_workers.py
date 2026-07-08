#!/usr/bin/python3
"""Wait until the MPI hostfile is complete and all workers accept SSH."""

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

DEFAULT_HOSTFILE = "/etc/mpi/hostfile"

SSH_BASE_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S UTC %Y")


def log(msg: str) -> None:
    print(f"{_ts()}\n{msg}", flush=True)


def read_hosts(hostfile: str) -> list[str]:
    hosts: list[str] = []
    with open(hostfile, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            host = line.split()[0]
            if host:
                hosts.append(host)
    return hosts


def wait_for_hostfile(hostfile: str, workers: int, retry_sleep: float = 10.0) -> list[str]:
    """Block until hostfile line count equals *workers*, then return hostnames."""
    while True:
        hosts = read_hosts(hostfile)
        if len(hosts) == workers:
            log("Ready")
            with open(hostfile, encoding="utf-8") as fh:
                sys.stdout.write(fh.read())
                sys.stdout.flush()
            return hosts
        log(f"not ready ... ({len(hosts)}/{workers} lines in {hostfile})")
        time.sleep(retry_sleep)


def ssh_reachable(host: str, connect_timeout: int = 5) -> bool:
    cmd = [
        "ssh",
        *SSH_BASE_OPTS,
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-n",
        host,
        "echo",
        host,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=connect_timeout + 15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def wait_for_host(host: str, retry_sleep: float = 10.0, connect_timeout: int = 5) -> None:
    """Retry SSH until *host* responds."""
    while not ssh_reachable(host, connect_timeout=connect_timeout):
        log(f"Pod {host} is not up ...")
        time.sleep(retry_sleep)
    log(f"Pod {host} is ready")


def wait_for_all_hosts(
    hosts: list[str],
    threads: int,
    retry_sleep: float = 10.0,
    connect_timeout: int = 5,
) -> None:
    """SSH reachability check for every host, limited to *threads* processes."""
    if not hosts:
        return
    processes = max(1, threads)
    with ProcessPoolExecutor(max_workers=processes) as pool:
        futures = [
            pool.submit(wait_for_host, host, retry_sleep, connect_timeout)
            for host in hosts
        ]
        for future in as_completed(futures):
            future.result()


def main() -> int:
    hostfile = os.getenv("HOSTFILE", DEFAULT_HOSTFILE)
    workers = int(os.environ["WORKERS"])
    threads = int(os.getenv("THREADS", "64"))
    retry_sleep = float(os.getenv("RETRY_SLEEP", "10"))
    connect_timeout = int(os.getenv("SSH_CONNECT_TIMEOUT", "5"))

    hosts = wait_for_hostfile(hostfile, workers, retry_sleep=retry_sleep)
    wait_for_all_hosts(
        hosts,
        threads=threads,
        retry_sleep=retry_sleep,
        connect_timeout=connect_timeout,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)