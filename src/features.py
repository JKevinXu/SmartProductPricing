"""Feature builders for SmartProductPricing models."""

from __future__ import annotations

import re
import string

import numpy as np
import pandas as pd


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_PACK_RE = re.compile(
    r"(?:pack\s+of\s+(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:pack|pk|count|ct|pcs|pieces)\b)",
    flags=re.IGNORECASE,
)
_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(oz|ounce|ounces|lb|lbs|pound|pounds|g|gram|grams|kg|ml|l|liter|liters|fl\s*oz)\b",
    flags=re.IGNORECASE,
)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(series: pd.Series) -> pd.Series:
    """Normalize catalog text while preserving useful numeric tokens."""
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_numeric_text_features(frame: pd.DataFrame) -> np.ndarray:
    """Create lightweight dense features from product catalog text."""
    text = clean_text(frame["catalog_content"])
    has_image = frame.get("image_link", pd.Series("", index=frame.index)).fillna("").astype(str)

    rows: list[list[float]] = []
    for value, image_value in zip(text, has_image, strict=False):
        words = value.split()
        numbers = [float(match) for match in _NUMBER_RE.findall(value)]
        pack_matches = _PACK_RE.findall(value)
        unit_matches = _UNIT_RE.findall(value)

        letters = [char for char in value if char.isalpha()]
        upper_letters = [char for char in letters if char.isupper()]
        punctuation_count = sum(1 for char in value if char in string.punctuation)
        stripped_punctuation = value.translate(_PUNCT_TABLE)

        pack_values = [
            float(first or second)
            for first, second in pack_matches
            if first or second
        ]
        unit_values = [float(amount) for amount, _unit in unit_matches]

        rows.append(
            [
                float(len(value)),
                float(len(words)),
                float(sum(char.isdigit() for char in value)),
                float(len(upper_letters) / max(len(letters), 1)),
                float(punctuation_count),
                float(len(numbers)),
                float(np.mean(numbers) if numbers else 0.0),
                float(np.max(numbers) if numbers else 0.0),
                float(np.max(pack_values) if pack_values else 1.0),
                float(np.max(unit_values) if unit_values else 0.0),
                float(bool(unit_values)),
                float(bool(pack_values)),
                float(bool(stripped_punctuation.strip())),
                float(bool(image_value.strip())),
            ]
        )

    return np.asarray(rows, dtype=np.float32)


def numeric_feature_names() -> list[str]:
    return [
        "char_count",
        "word_count",
        "digit_count",
        "uppercase_ratio",
        "punctuation_count",
        "number_count",
        "number_mean",
        "number_max",
        "pack_count_guess",
        "unit_value_guess",
        "has_unit",
        "has_pack",
        "has_text",
        "has_image_link",
    ]
