"""Artifact storage interface and atomic local implementation."""
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
import os


class Storage(Protocol):
    def read(self, artifact_id: str) -> bytes: ...
    def write(self, artifact_id: str, content: bytes) -> None: ...


class LocalStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _path(self, artifact_id: str) -> Path:
        path = (self.root / artifact_id).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Artifact path must be inside its storage root")
        return path

    def read(self, artifact_id: str) -> bytes:
        return self._path(artifact_id).read_bytes()

    def write(self, artifact_id: str, content: bytes) -> None:
        path = self._path(artifact_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temp_path = Path(temporary.name)
            try:
                temporary.write(content)
                temporary.close()
                os.replace(temp_path, path)
            finally:
                temp_path.unlink(missing_ok=True)
