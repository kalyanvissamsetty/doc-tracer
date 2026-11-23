"""Pair pages by identifiers and content, then use order for equal-sized gaps."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pymupdf
from scipy.optimize import linear_sum_assignment

from redline.models import PagePair

SHEET = re.compile(r"\b(?:sheet|drawing|dwg)\s*(?:no\.?|number|#)?\s*[:#-]?\s+([a-z0-9][a-z0-9.-]*)\b", re.I)
NUMBER = re.compile(r"\bpage\s+(\d+)\s*(?:of|/)\s*\d+\b", re.I)
JOB = re.compile(r"\b(?:job|project|report|order|po)\s*(?:no\.?|number|#)?\s*[:#-]?\s+([a-z0-9][a-z0-9.-]*)\b", re.I)


@dataclass
class PageIdentity:
    keys: set[tuple[str, str]]
    words: set[str]


def _identities(path: str | Path) -> list[PageIdentity]:
    identities = []
    with pymupdf.open(path, filetype="pdf") as pdf:
        for page in pdf:
            text = page.get_text(sort=True).casefold()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            keys = set()
            if lines:
                keys.add(("title", " ".join(lines[0].split())))
            for category, pattern in (("sheet", SHEET), ("number", NUMBER), ("job", JOB)):
                keys.update((category, value) for value in pattern.findall(text))
            for image in page.get_image_info(hashes=True):
                if pymupdf.Rect(image["bbox"]).get_area() >= 0.5 * page.rect.get_area():
                    keys.add(("image", image["digest"].hex()))
            identities.append(PageIdentity(keys, set(re.findall(r"\w+", text))))
    return identities


def pair_pages(v1: str | Path, v2: str | Path) -> list[PagePair]:
    old = _identities(v1)
    new = _identities(v2)
    old_counts = Counter(key for page in old for key in page.keys)
    new_counts = Counter(key for page in new for key in page.keys)
    weights = {"sheet": 10, "image": 9, "job": 5, "title": 4, "number": 2}
    scores = np.zeros((len(old), len(new)))
    for i, left in enumerate(old):
        for j, right in enumerate(new):
            for key in left.keys & right.keys:
                if old_counts[key] == new_counts[key] == 1:
                    scores[i, j] += weights[key[0]]
            if left.words and right.words:
                similarity = len(left.words & right.words) / len(left.words | right.words)
                if similarity >= 0.55:
                    scores[i, j] += 3 * similarity
    rows, cols = linear_sum_assignment(scores, maximize=True)
    paired = {int(i): int(j) for i, j in zip(rows, cols) if scores[i, j] > 0}

    # A positional fallback is only used within equally sized, ordered gaps.
    anchors = [(-1, -1), *sorted(paired.items()), (len(old), len(new))]
    reserved = set(paired.values())
    for (i, j), (next_i, next_j) in zip(anchors, anchors[1:]):
        if next_i - i == next_j - j and not any(k in reserved for k in range(j + 1, next_j)):
            paired.update(zip(range(i + 1, next_i), range(j + 1, next_j)))

    by_new = {j: i + 1 for i, j in paired.items()}
    return [
        *(PagePair(v1_page=by_new.get(j), v2_page=j + 1) for j in range(len(new))),
        *(PagePair(v1_page=i + 1, v2_page=None) for i in range(len(old)) if i not in paired),
    ]
