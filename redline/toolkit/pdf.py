"""PDF inspection and rendering. Rendered PNGs respect displayed page rotation."""
from hashlib import sha256
from math import isfinite
from pathlib import Path

import pymupdf

from redline.models import Annotation, PageProbe, ProbeReport
from redline.storage import Storage
from redline.toolkit.images import embedded_images


def probe_pdf(path: str | Path) -> ProbeReport:
    pages = []
    with pymupdf.open(path, filetype="pdf") as pdf:
        for page in pdf:
            bounds = page.rect * page.derotation_matrix
            images = embedded_images(page)
            annotations = _page_annotations(page, include_content=False)
            pages.append(PageProbe(
                page=page.number + 1, width=bounds.width, height=bounds.height,
                rotation=page.rotation, has_text=bool(page.get_text().strip()), images=images,
                vector_count=len(page.get_cdrawings()), annotations=annotations,
                full_page_scan=any(
                    (pymupdf.Rect(image.bbox) & bounds).get_area() >= 0.9 * bounds.get_area()
                    for image in images
                ),
            ))
    return ProbeReport(page_count=len(pages), pages=pages)


def render_page(
    path: str | Path, page: int, dpi: float, annots: bool = True, *, storage: Storage
) -> str:
    if not isfinite(dpi) or dpi <= 0:
        raise ValueError("DPI must be finite and positive.")
    with pymupdf.open(path, filetype="pdf") as pdf:
        if not 1 <= page <= len(pdf):
            raise ValueError(f"Page must be between 1 and {len(pdf)}.")
        content = pdf[page - 1].get_pixmap(
            matrix=pymupdf.Matrix(dpi / 72, dpi / 72), annots=annots,
        ).tobytes("png")
    image_id = f"renders/{sha256(content).hexdigest()}.png"
    storage.write(image_id, content)
    return image_id


def _page_annotations(page: pymupdf.Page, *, include_content: bool = True) -> list[Annotation]:
    annotations = [
        Annotation(
            kind=annot.type[1], bbox=tuple(annot.rect),
            content=(annot.info.get("content") or None) if include_content else None,
        )
        for annot in page.annots() or ()
    ]
    for widget in page.widgets() or ():
        value = widget.field_value if include_content else None
        if widget.field_type in (pymupdf.PDF_WIDGET_TYPE_CHECKBOX, pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON):
            value = None if value in (False, "Off") else value
        annotations.append(Annotation(
            kind=f"Widget: {widget.field_type_string}", bbox=tuple(widget.rect),
            field_name=widget.field_name if include_content else None,
            content=str(value) if value is not None else None,
        ))
    annotations.extend(
        Annotation(kind="Link", bbox=tuple(link["from"] * page.derotation_matrix))
        for link in page.get_links()
    )
    return annotations


def split_annotation_layer(
    path: str | Path, page: int, *, storage: Storage, dpi: float = 144
) -> tuple[str, list[Annotation]]:
    """Render PDF content without overlay annotations and return annotation metadata.

    Flattened review marks cannot be removed by this operation.
    """
    image_id = render_page(path, page, dpi, annots=False, storage=storage)
    with pymupdf.open(path, filetype="pdf") as pdf:
        return image_id, _page_annotations(pdf[page - 1])
