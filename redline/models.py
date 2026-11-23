"""Page numbers are one-based. PDF boxes use unrotated PyMuPDF page points."""
from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]


class EmbeddedImage(BaseModel):
    bbox: BBox
    digest: str
    width: int
    height: int


class Annotation(BaseModel):
    kind: str
    bbox: BBox
    content: str | None = None
    field_name: str | None = None


class PageProbe(BaseModel):
    page: int
    width: float
    height: float
    rotation: int
    has_text: bool
    images: list[EmbeddedImage]
    vector_count: int
    annotations: list[Annotation]
    full_page_scan: bool


class ProbeReport(BaseModel):
    page_count: int
    pages: list[PageProbe]


class PagePair(BaseModel):
    v1_page: int | None
    v2_page: int | None


class Region(BaseModel):
    bbox: BBox
    area: int = Field(ge=1)


class Change(BaseModel):
    id: int = Field(ge=1)
    type: Literal["addition", "deletion", "modification"]
    category: Literal["value", "row", "page", "stamp", "image", "review_markup", "cosmetic"]
    field: str
    v1_value: str | None = None
    v2_value: str | None = None
    bbox_v2: BBox | None = None
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    review_request_status: str | None = None
    human_status: Literal["pending", "accepted", "rejected", "edited"] = "pending"

    @property
    def needs_review(self) -> bool:
        return self.confidence < 0.7


class PageChanges(PagePair):
    method: Literal["identical", "pixel_diff", "structural", "vector", "image_compare", "text"]
    changes: list[Change] = Field(default_factory=list)


class Cost(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    docintel_pages: int = Field(default=0, ge=0)
    estimated_usd: float = Field(default=0, ge=0)


class GeometryEdit(BaseModel):
    change_id: int = Field(ge=1)
    v2_page: int = Field(ge=1)
    before: BBox
    after: BBox
    row: int = Field(ge=1)
    column: int = Field(ge=1)
    actor: Literal["agent", "reviewer"]


class ChangeLog(BaseModel):
    job_id: str
    v1: str
    v2: str
    model_deployments: dict[str, str] = Field(default_factory=dict)
    cost: Cost = Field(default_factory=Cost)
    pages: list[PageChanges] = Field(default_factory=list)
    geometry_edits: list[GeometryEdit] = Field(default_factory=list)
    comment_only_tags: dict[int, list[int]] = Field(default_factory=dict)
