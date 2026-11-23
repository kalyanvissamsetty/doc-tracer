"""Read local render artifacts or crop them in displayed pixel coordinates."""
from io import BytesIO
from math import ceil, floor, isfinite

from PIL import Image

from redline.models import BBox
from redline.storage import Storage


def view_page(image_id: str, *, storage: Storage) -> bytes:
    """Return the PNG bytes from a render artifact."""
    return storage.read(image_id)


def view_crop(image_id: str, bbox: BBox, *, storage: Storage) -> bytes:
    """Return a PNG crop. Coordinates are pixels and the right/bottom edges are exclusive."""
    x0, y0, x1, y1 = bbox
    if not all(map(isfinite, bbox)) or x0 >= x1 or y0 >= y1:
        raise ValueError("Crop bounds must be finite and nonempty.")
    with Image.open(BytesIO(view_page(image_id, storage=storage))) as image:
        bounds = max(0, floor(x0)), max(0, floor(y0)), min(image.width, ceil(x1)), min(image.height, ceil(y1))
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("Crop is outside the rendered page.")
        output = BytesIO()
        image.crop(bounds).save(output, format="PNG")
        return output.getvalue()
