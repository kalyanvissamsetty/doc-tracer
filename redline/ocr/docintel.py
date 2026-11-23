"""Document Intelligence Read and Layout with content-addressed caching."""
from collections.abc import Callable
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.exceptions import AzureError

from redline.config import Settings
from redline.storage import Storage

API_VERSION = "2024-11-30"


class OCRRequestError(RuntimeError):
    pass


class DocumentIntel:
    def __init__(
        self, settings: Settings, storage: Storage, *, client: DocumentIntelligenceClient | None = None,
        retry_total: int = 3, credential: TokenCredential | None = None,
    ):
        self.storage = storage
        self.endpoint_id = sha256(settings.azure_docintel_endpoint.encode()).hexdigest()[:16]
        self.client = client if client is not None else DocumentIntelligenceClient(
            endpoint=settings.azure_docintel_endpoint,
            credential=credential if credential is not None else AzureKeyCredential(settings.azure_docintel_key.get_secret_value()),
            api_version=API_VERSION, retry_total=retry_total,
        )

    def extract(
        self, path: str | Path, page: int, *, mode: Literal["layout", "read"] = "layout",
        before_request: Callable[[], None] | None = None,
    ) -> dict:
        if page < 1:
            raise ValueError("Page numbers start at 1.")
        if mode not in ("layout", "read"):
            raise ValueError("OCR mode must be layout or read.")
        with Path(path).open("rb") as pdf:
            digest = sha256()
            for chunk in iter(lambda: pdf.read(1024 * 1024), b""):
                digest.update(chunk)
            cache_id = f"ocr/{API_VERSION}/{self.endpoint_id}/{digest.hexdigest()}/{mode}-{page}.json"
            try:
                return json.loads(self.storage.read(cache_id))
            except (FileNotFoundError, ValueError):
                pass
            if before_request is not None:
                before_request()
            pdf.seek(0)
            options = {"features": [DocumentAnalysisFeature.KEY_VALUE_PAIRS]} if mode == "layout" else {}
            try:
                result = self.client.begin_analyze_document(
                    f"prebuilt-{mode}", body=pdf, content_type="application/octet-stream",
                    pages=str(page), **options,
                ).result().as_dict()
            except AzureError as exc:
                status = getattr(exc, "status_code", None)
                detail = f" (HTTP {status})" if status else ""
                raise OCRRequestError(f"Document Intelligence request failed{detail}.") from None
        self.storage.write(cache_id, json.dumps(result).encode())
        return result
