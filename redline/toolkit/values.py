"""Normalize comparison text while retaining distinctions between unlike values."""
from datetime import date, datetime
import re
import unicodedata

QUOTES = str.maketrans({"‘": "'", "’": "'", "′": "'", "“": '"', "”": '"', "″": '"', "−": "-"})
OCR_DIGITS = str.maketrans({"o": "0", "l": "1", "s": "5"})
UNITS = {
    "in": "in", "inch": "in", "inches": "in", '"': "in", "''": "in",
    "ft": "ft", "foot": "ft", "feet": "ft", "'": "ft",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "psi": "psi", "kpa": "kpa", "mpa": "mpa", "bar": "bar",
    "mm": "mm", "cm": "cm", "m": "m", "kg": "kg",
    "minutes": "min", "minute": "min", "min": "min",
    "hours": "h", "hour": "h", "hrs": "h", "hr": "h", "h": "h",
}
UNIT_PATTERN = "|".join(re.escape(unit) for unit in sorted(UNITS, key=len, reverse=True))
MEASUREMENT = re.compile(rf"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*({UNIT_PATTERN})(?![a-z])")


def _normalize_date(value: str) -> str:
    iso = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", value)
    numeric = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", value)
    try:
        if iso:
            return date(*map(int, iso.groups())).isoformat()
        if numeric:
            first, second, year = map(int, numeric.groups())
            if first > 12 or second > 12 or first == second:
                month, day = (second, first) if first > 12 else (first, second)
                return date(year, month, day).isoformat()
        for pattern in ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(value, pattern).date().isoformat()
            except ValueError:
                continue
    except ValueError:
        pass
    return value


def normalize_value(text: str, *, numeric_ocr: bool = False) -> str:
    """Fold formatting. Enable OCR substitutions only for known numeric fields."""
    value = unicodedata.normalize("NFKC", text).translate(QUOTES).casefold()
    value = " ".join(value.split())
    value = re.sub(r"#\s+", "#", value)
    if numeric_ocr:
        value = re.sub(
            r"(?<!\w)[0-9ols]+(?:\.[0-9ols]+)?(?!\w)",
            lambda match: match[0].translate(OCR_DIGITS)
            if match[0] == value or len(match[0]) == 1 or "." in match[0] or any(char.isdigit() for char in match[0])
            else match[0],
            value,
        )
    value = _normalize_date(value)
    return MEASUREMENT.sub(lambda match: f"{match[1]} {UNITS[match[2]]}", value)
