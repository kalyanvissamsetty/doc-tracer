"""Fail closed before executing input if Docker did not apply the isolation limits."""
import os
from pathlib import Path
import sys

CGROUP = Path("/sys/fs/cgroup")


def limited(cgroup: Path = CGROUP):
    try:
        cpu, period = (cgroup / "cpu.max").read_text().split()
        return (
            os.getuid() == 65534
            and int(cpu) * 2 <= int(period)
            and int((cgroup / "memory.max").read_text()) <= 128 * 1024 * 1024
            and int((cgroup / "memory.swap.max").read_text()) == 0
            and int((cgroup / "pids.max").read_text()) <= 16
        )
    except (OSError, ValueError):
        return False


def main():
    if not limited():
        sys.exit(126)
    source = sys.stdin.read(16 * 1024 + 1)
    if not source or len(source.encode("utf-8")) > 16 * 1024:
        sys.exit(125)
    namespace = {"__name__": "__main__"}
    exec(compile(source, "<sandbox>", "exec"), namespace)


if __name__ == "__main__":
    main()
