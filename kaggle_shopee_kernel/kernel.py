from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


KAGGLE_INPUT_DIR = Path("/kaggle/input/shopee-product-matching")
LOCAL_INPUT_DIR = Path("data/shopee/raw")
INPUT_DIR = KAGGLE_INPUT_DIR if KAGGLE_INPUT_DIR.exists() else LOCAL_INPUT_DIR
OUTPUT_PATH = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else Path("data/submissions/shopee_kernel_submission.csv")

CHAR_TITLE_THRESHOLD = 0.72
WORD_TITLE_THRESHOLD = 0.66
MAX_CHAR_FEATURES = 100000
MAX_WORD_FEATURES = 100000
MAX_CHAR_TITLE_NEIGHBORS = 80
MAX_WORD_TITLE_NEIGHBORS = 80

PUBLIC_TEST_FALLBACK = pd.DataFrame(
    {
        "posting_id": [
            "test_2255846744",
            "test_3588702337",
            "test_4015706929",
        ],
        "image_phash": [
            "ecc292392dc7687a",
            "e9968f60d2699e2c",
            "ba81c17e3581cabe",
        ],
        "title": [
            "Edufuntoys - CHARACTER PHONE ada lampu dan musik/ mainan telepon",
            "(Beli 1 Free Spatula) Masker Komedo | Blackheads Mask 10gr by Flawless Go Surabaya | Flawless.Go",
            "READY Lemonilo Mie instant sehat kuah dan goreng",
        ],
    }
)


def normalize_titles(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", "missing")
    )


def exact_phash_matches(frame: pd.DataFrame) -> dict[str, set[str]]:
    if "image_phash" not in frame.columns:
        return {str(posting_id): {str(posting_id)} for posting_id in frame["posting_id"]}

    groups = (
        frame.groupby("image_phash", dropna=False)["posting_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    return {
        str(row.posting_id): set(groups.get(row.image_phash, {str(row.posting_id)}))
        for row in frame[["posting_id", "image_phash"]].itertuples(index=False)
    }


def tfidf_neighbor_matches(
    frame: pd.DataFrame,
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    threshold: float,
    max_features: int,
    max_neighbors: int,
) -> dict[str, set[str]]:
    titles = normalize_titles(frame["title"])
    vectorizer_args = {
        "analyzer": analyzer,
        "ngram_range": ngram_range,
        "min_df": 1,
        "max_features": max_features,
        "lowercase": False,
        "dtype": np.float32,
    }
    if analyzer == "word":
        vectorizer_args["token_pattern"] = r"(?u)\b\w+\b"

    vectorizer = TfidfVectorizer(
        **vectorizer_args,
    )
    title_matrix = vectorizer.fit_transform(titles)
    neighbor_count = min(max_neighbors, len(frame))
    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(title_matrix)
    distances, indices = model.kneighbors(title_matrix, return_distance=True)

    posting_ids = frame["posting_id"].astype(str).to_numpy()
    matches: dict[str, set[str]] = {}
    for row_index, posting_id in enumerate(posting_ids):
        row_matches = {posting_id}
        similarities = 1.0 - distances[row_index]
        for similarity, neighbor_index in zip(similarities, indices[row_index], strict=False):
            if similarity >= threshold:
                row_matches.add(str(posting_ids[neighbor_index]))
        matches[str(posting_id)] = row_matches
    return matches


def char_title_matches(frame: pd.DataFrame) -> dict[str, set[str]]:
    return tfidf_neighbor_matches(
        frame,
        analyzer="char_wb",
        ngram_range=(3, 5),
        threshold=CHAR_TITLE_THRESHOLD,
        max_features=MAX_CHAR_FEATURES,
        max_neighbors=MAX_CHAR_TITLE_NEIGHBORS,
    )


def word_title_matches(frame: pd.DataFrame) -> dict[str, set[str]]:
    return tfidf_neighbor_matches(
        frame,
        analyzer="word",
        ngram_range=(1, 2),
        threshold=WORD_TITLE_THRESHOLD,
        max_features=MAX_WORD_FEATURES,
        max_neighbors=MAX_WORD_TITLE_NEIGHBORS,
    )


def combine_matches(*match_maps: dict[str, set[str]]) -> dict[str, set[str]]:
    combined: dict[str, set[str]] = {}
    for match_map in match_maps:
        for posting_id, matches in match_map.items():
            combined.setdefault(posting_id, {posting_id}).update(matches)
    return combined


def find_test_csv() -> Path | None:
    candidates = []
    for root in [KAGGLE_INPUT_DIR, Path("/kaggle/input"), LOCAL_INPUT_DIR]:
        if root.exists():
            candidates.extend(root.rglob("test.csv"))

    for candidate in candidates:
        try:
            columns = pd.read_csv(candidate, nrows=0).columns
        except Exception:
            continue
        if {"posting_id", "title"}.issubset(set(columns)):
            return candidate
    return None


def main() -> None:
    test_path = find_test_csv()
    if test_path is None:
        print("Could not find mounted test.csv; using public fallback rows.")
        test = PUBLIC_TEST_FALLBACK.copy()
    else:
        print(f"Using test CSV: {test_path}")
        test = pd.read_csv(test_path)
    predictions = combine_matches(
        exact_phash_matches(test),
        char_title_matches(test),
        word_title_matches(test),
    )
    submission = pd.DataFrame(
        {
            "posting_id": test["posting_id"].astype(str),
            "matches": [
                " ".join(sorted(predictions[str(posting_id)]))
                for posting_id in test["posting_id"].astype(str)
            ],
        }
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} with {len(submission)} rows")


if __name__ == "__main__":
    main()
