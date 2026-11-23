"""Private Blob Storage artifact backend and commit-last archive manifests."""
from collections.abc import Mapping
from json import dumps
from pathlib import Path
from typing import TYPE_CHECKING

from azure.core.exceptions import AzureError, ResourceNotFoundError
if TYPE_CHECKING:
    from azure.storage.blob import ContainerClient

from redline.storage import Storage


class ArtifactArchiveError(RuntimeError):
    pass


def _key(value: str) -> str:
    if "\\" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Artifact identifiers must be relative paths without traversal.")
    return value


class BlobStorage:
    def __init__(self, container: "ContainerClient", prefix: str = "redline"):
        self.container = container
        self.prefix = _key(prefix)

    def _blob(self, artifact_id: str):
        return self.container.get_blob_client(f"{self.prefix}/{_key(artifact_id)}")

    def read(self, artifact_id: str) -> bytes:
        try:
            return self._blob(artifact_id).download_blob().readall()
        except ResourceNotFoundError:
            raise FileNotFoundError("Artifact was not found in Blob Storage.") from None
        except AzureError:
            raise ArtifactArchiveError("Could not read the Blob Storage artifact.") from None

    def write(self, artifact_id: str, content: bytes) -> None:
        try:
            self._blob(artifact_id).upload_blob(content, overwrite=True)
        except AzureError:
            raise ArtifactArchiveError("Could not write the Blob Storage artifact.") from None

    def close(self) -> None:
        self.container.close()


def archive_artifacts(store: Storage, prefix: str, paths: Mapping[str, Path]) -> None:
    """Publish the manifest only after every member has uploaded successfully."""
    for name, source in paths.items():
        store.write(f"{prefix}/{name}", source.read_bytes())
    store.write(f"{prefix}/manifest.json", dumps({"artifacts": sorted(paths)}).encode())
