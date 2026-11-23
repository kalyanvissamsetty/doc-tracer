"""Run and score golden PDF comparisons by document type."""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal

import numpy as np
import pymupdf
from pydantic import BaseModel, Field, model_validator
from scipy.optimize import linear_sum_assignment

from redline.comparison import ComparisonOptions, compare
from redline.models import BBox, ChangeLog
from redline.toolkit import normalize_value
from redline.tracing import JsonlTracer


class ExpectedChange(BaseModel):
    v1_page: int | None = Field(default=None, ge=1)
    v2_page: int | None = Field(default=None, ge=1)
    type: Literal["addition", "deletion", "modification"]
    bbox_v2: BBox | None = None
    v1_value: str | None = None
    v2_value: str | None = None

    @model_validator(mode="after")
    def check_location(self):
        if self.v2_page is None and (self.v1_page is None or self.type != "deletion"):
            raise ValueError("An expected change needs a V2 page, or a V1 page for a deleted page.")
        if self.bbox_v2 is not None and (self.v2_page is None or pymupdf.Rect(self.bbox_v2).is_empty):
            raise ValueError("An expected box needs a V2 page and positive area.")
        return self


class GoldenCase(BaseModel):
    document_type: str = Field(min_length=1)
    changes: list[ExpectedChange]


@dataclass
class Stats:
    expected: int = 0
    predicted: int = 0
    correct: int = 0
    localized: int = 0
    localized_type_correct: int = 0
    values_checked: int = 0
    values_correct: int = 0
    pages: int = 0
    seconds: float = 0
    usd: float = 0

    def add(self, other: "Stats") -> None:
        for name, value in asdict(other).items():
            setattr(self, name, getattr(self, name) + value)

    def metrics(self) -> dict:
        def ratio(num, denom):
            return num / denom if denom else None

        return {
            **asdict(self), "recall": ratio(self.correct, self.expected),
            "precision": ratio(self.correct, self.predicted),
            "type_accuracy": ratio(self.localized_type_correct, self.localized),
            "value_accuracy": ratio(self.values_correct, self.values_checked),
            "seconds_per_page": ratio(self.seconds, self.pages),
            "usd_per_page": ratio(self.usd, self.pages),
        }


def _iou(a: BBox | None, b: BBox | None) -> float:
    if a is None or b is None:
        return float(a is None and b is None)
    ar, br = pymupdf.Rect(a), pymupdf.Rect(b)
    common = (ar & br).get_area()
    union = ar.get_area() + br.get_area() - common
    return common / union if union else 0.0


def _match(scores: np.ndarray) -> list[tuple[int, int]]:
    # Prioritize match count before overlap to keep scoring one-to-one.
    rows, columns = linear_sum_assignment(np.where(scores > 0, scores + min(scores.shape), 0), maximize=True)
    return [(int(i), int(j)) for i, j in zip(rows, columns) if scores[i, j] > 0]


def score(log: ChangeLog, golden: GoldenCase, *, pages: int, elapsed: float) -> Stats:
    predictions = [
        (group.v1_page, group.v2_page, change)
        for group in log.pages for change in group.changes if change.category != "cosmetic"
    ]
    expected = golden.changes
    geometry = np.zeros((len(expected), len(predictions)))
    type_matches = np.zeros(geometry.shape, dtype=bool)
    for i, target in enumerate(expected):
        for j, (v1, v2, change) in enumerate(predictions):
            if v2 != target.v2_page or (v2 is None and v1 != target.v1_page):
                continue
            overlap = _iou(target.bbox_v2, change.bbox_v2)
            if overlap >= 0.3:
                geometry[i, j] = overlap
                type_matches[i, j] = target.type == change.type
    typed = _match(geometry * type_matches)
    localized = _match(geometry + (geometry > 0) * type_matches * 1e-7)
    result = Stats(
        expected=len(expected), predicted=len(predictions), correct=len(typed), localized=len(localized),
        localized_type_correct=sum(bool(type_matches[i, j]) for i, j in localized), pages=pages,
        seconds=elapsed, usd=log.cost.estimated_usd,
    )
    for i, j in typed:
        labels = expected[i]
        prediction = predictions[j][2]
        for key in labels.model_fields_set & {"v1_value", "v2_value"}:
            result.values_checked += 1
            result.values_correct += normalize_value(getattr(labels, key) or "") == normalize_value(
                getattr(prediction, key) or ""
            )
    return result


def run_suite(golden_dir: Path, predictions_dir: Path | None = None) -> dict:
    manifests = sorted(golden_dir.glob("*/expected_changes.json"))
    if not manifests:
        raise ValueError("No golden cases were found.")
    totals = defaultdict(Stats)
    results = []
    for manifest in manifests:
        case = manifest.parent
        expected = GoldenCase.model_validate_json(manifest.read_bytes())
        v1, v2 = case / "v1.pdf", case / "v2.pdf"
        with pymupdf.open(v1) as original, pymupdf.open(v2) as revised:
            pages = max(len(original), len(revised))
        elapsed = None
        if predictions_dir is not None:
            log = ChangeLog.model_validate_json((predictions_dir / case.name / "changes.json").read_bytes())
        else:
            with TemporaryDirectory(prefix="redline-eval-") as workspace:
                root = Path(workspace)
                start = perf_counter()
                result = compare(
                    v1, v2, root=root, options=ComparisonOptions(
                        max(60, 2 * pages + 6), Decimal(5), document_type=expected.document_type,
                    ),
                    tracer=JsonlTracer(root / "trace.jsonl"),
                )
                elapsed = perf_counter() - start
                log = ChangeLog.model_validate_json(result.changes.read_bytes())
        stats = score(log, expected, pages=pages, elapsed=elapsed or 0)
        totals[expected.document_type].add(stats)
        metrics = stats.metrics()
        if elapsed is None:
            metrics["seconds"] = metrics["seconds_per_page"] = None
        results.append({"case": case.name, "document_type": expected.document_type, **metrics})

    groups = {kind: stats.metrics() for kind, stats in sorted(totals.items())}
    if predictions_dir is not None:
        for metrics in groups.values():
            metrics["seconds"] = metrics["seconds_per_page"] = None
    return {"by_document_type": groups, "cases": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate PDF comparisons against labeled golden cases.")
    parser.add_argument("--golden-dir", type=Path, default=Path("tests/golden"))
    parser.add_argument("--predictions-dir", type=Path, help="Score existing <case>/changes.json files")
    args = parser.parse_args(argv)
    try:
        report = run_suite(args.golden_dir, args.predictions_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Evaluation failed ({type(exc).__name__}). Check the suite and prediction files.", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
