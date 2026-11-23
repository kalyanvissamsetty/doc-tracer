"""Deterministic PDF inspection, comparison, and markup tools."""
from redline.toolkit.images import compare_embedded_images
from redline.toolkit.markup import write_markup
from redline.toolkit.pairing import pair_pages
from redline.toolkit.pdf import probe_pdf, render_page, split_annotation_layer
from redline.toolkit.scans import align_images, compare_scanned_drawing
from redline.toolkit.vector import vector_diff
from redline.toolkit.stamps import detect_stamps
from redline.toolkit.tables import detect_table_grid, extract_table, match_rows
from redline.toolkit.values import normalize_value
from redline.toolkit.views import view_crop, view_page
from redline.toolkit.visual import pixel_diff

__all__ = [
    "align_images", "compare_scanned_drawing", "vector_diff", "compare_embedded_images", "detect_stamps", "detect_table_grid", "extract_table",
    "match_rows", "normalize_value", "pair_pages", "pixel_diff", "probe_pdf",
    "render_page", "split_annotation_layer", "view_crop", "view_page", "write_markup",
]
