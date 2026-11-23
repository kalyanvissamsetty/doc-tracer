"""Run optional document calculations in a fixed, isolated Docker container."""
from dataclasses import dataclass
import os
import selectors
import subprocess
from tempfile import TemporaryFile
from time import monotonic
from uuid import uuid4

from redline.tracing import Tracer

IMAGE = "redline-python-sandbox:local"
MAX_CODE = 16 * 1024
MAX_OUTPUT = 12 * 1024
TIMEOUT_SECONDS = 8


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int


def _command(name: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--interactive", "--pull=never", "--name", name,
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--cpus=0.5", "--memory=128m", "--memory-swap=128m", "--pids-limit=16", "--cgroupns=private",
        "--ulimit", "nofile=64:64", "--user=65534:65534", "--ipc=none",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m", "--log-driver=none",
        IMAGE,
    ]


def _collect(process: subprocess.Popen, deadline: float) -> SandboxResult:
    data = {process.stdout: bytearray(), process.stderr: bytearray()}
    with selectors.DefaultSelector() as selector:
        for pipe in data:
            selector.register(pipe, selectors.EVENT_READ)
        size = 0
        while selector.get_map():
            remaining = deadline - monotonic()
            if remaining <= 0 or not (ready := selector.select(remaining)):
                raise SandboxError("The Python sandbox timed out.")
            for key, _ in ready:
                pipe = key.fileobj
                chunk = os.read(pipe.fileno(), 4096)
                if not chunk:
                    selector.unregister(pipe)
                    continue
                size += len(chunk)
                if size > MAX_OUTPUT:
                    raise SandboxError("The Python sandbox exceeded its output limit.")
                data[pipe].extend(chunk)
    try:
        code = process.wait(timeout=max(0, deadline - monotonic()))
    except subprocess.TimeoutExpired:
        raise SandboxError("The Python sandbox timed out.") from None
    if code in {124, 137} and monotonic() >= deadline - 2:
        raise SandboxError("The Python sandbox exceeded its runtime limit.")
    if code in {125, 126, 127}:
        raise SandboxError("The sandbox requires Docker with enforced cgroup v2 CPU, memory, swap, and process limits.")
    return SandboxResult(*(data[pipe].decode("utf-8", errors="replace") for pipe in data), code)


def run_python(code: str, *, tracer: Tracer) -> SandboxResult:
    """Execute pure calculations with no host mounts, network, or inherited Python environment."""
    start = monotonic()
    status = "rejected"
    name = f"redline-sandbox-{uuid4().hex}"
    process = None
    try:
        content = code.encode("utf-8")
        if not content or len(content) > MAX_CODE:
            raise SandboxError("Supply sandbox code within the input size limit.")
        if os.name == "nt":
            raise SandboxError("Run the Python fallback from a Linux or macOS host, or inside WSL.")
        with TemporaryFile() as source:
            source.write(content)
            source.seek(0)
            try:
                process = subprocess.Popen(_command(name), stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except OSError:
                raise SandboxError("Docker is unavailable for the Python sandbox.") from None
            status = "stopped"
            try:
                result = _collect(process, start + TIMEOUT_SECONDS)
            finally:
                if process.poll() is None:
                    try:
                        subprocess.run(["docker", "rm", "-f", name], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=3, check=False)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=3)
                process.stdout.close()
                process.stderr.close()
        status = "completed" if result.exit_code == 0 else "failed"
        return result
    finally:
        tracer.record("sandbox_execution", status=status, duration_ms=round((monotonic() - start) * 1000))
