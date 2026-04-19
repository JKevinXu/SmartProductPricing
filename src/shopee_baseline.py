"""Build a Shopee Product Matching baseline submission.

The competition asks for a space-separated list of matching `posting_id`
values for every test row. This baseline combines exact image perceptual hash
matches with title TF-IDF nearest neighbors.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


@dataclass
class ShopeeConfig:
    train_csv: str
    test_csv: str
    output_path: str
    reports_dir: str
    title_threshold: float
    max_features: int
    max_title_neighbors: int
    min_df: int
    analyzer: str
    ngram_min: int
    ngram_max: int
    diagnostics_limit: int
    seed: int


def parse_args() -> ShopeeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/shopee/raw/train.csv")
    parser.add_argument("--test-csv", default="data/shopee/raw/test.csv")
    parser.add_argument("--output-path", default="data/submissions/shopee_submission.csv")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--title-threshold", type=float, default=0.72)
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--max-title-neighbors", type=int, default=80)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--analyzer", choices=["word", "char_wb"], default="char_wb")
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--diagnostics-limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return ShopeeConfig(**vars(args))


def validate_columns(frame: pd.DataFrame, required_columns: set[str], path: Path) -> None:
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")


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
        return {posting_id: {posting_id} for posting_id in frame["posting_id"]}

    groups = (
        frame.groupby("image_phash", dropna=False)["posting_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    return {
        str(row.posting_id): set(groups.get(row.image_phash, {str(row.posting_id)}))
        for row in frame[["posting_id", "image_phash"]].itertuples(index=False)
    }


def title_neighbor_matches(frame: pd.DataFrame, config: ShopeeConfig) -> dict[str, set[str]]:
    titles = normalize_titles(frame["title"])
    vectorizer = TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_df,
        max_features=config.max_features,
        lowercase=False,
        dtype=np.float32,
    )
    title_matrix = vectorizer.fit_transform(titles)
    neighbor_count = min(config.max_title_neighbors, len(frame))
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
            if similarity >= config.title_threshold:
                row_matches.add(str(posting_ids[neighbor_index]))
        matches[str(posting_id)] = row_matches
    return matches


def combine_matches(*match_maps: dict[str, set[str]]) -> dict[str, set[str]]:
    combined: dict[str, set[str]] = {}
    for match_map in match_maps:
        for posting_id, matches in match_map.items():
            combined.setdefault(posting_id, {posting_id}).update(matches)
    return combined


def true_matches_from_labels(frame: pd.DataFrame) -> dict[str, set[str]]:
    validate_columns(frame, {"posting_id", "label_group"}, Path("train_csv"))
    groups = frame.groupby("label_group")["posting_id"].apply(lambda values: set(values.astype(str)))
    return {
        str(row.posting_id): set(groups.loc[row.label_group])
        for row in frame[["posting_id", "label_group"]].itertuples(index=False)
    }


def f1_score_sets(y_true: set[str], y_pred: set[str]) -> float:
    intersection = len(y_true.intersection(y_pred))
    if intersection == 0:
        return 0.0
    return 2.0 * intersection / (len(y_true) + len(y_pred))


def evaluate_matches(frame: pd.DataFrame, predictions: dict[str, set[str]]) -> dict[str, float]:
    if "label_group" not in frame.columns:
        return {}
    truth = true_matches_from_labels(frame)
    scores = [
        f1_score_sets(truth[str(posting_id)], predictions[str(posting_id)])
        for posting_id in frame["posting_id"].astype(str)
    ]
    predicted_lengths = [len(predictions[str(posting_id)]) for posting_id in frame["posting_id"].astype(str)]
    return {
        "mean_f1": float(np.mean(scores)),
        "median_f1": float(np.median(scores)),
        "mean_match_count": float(np.mean(predicted_lengths)),
        "median_match_count": float(np.median(predicted_lengths)),
    }


def format_submission(frame: pd.DataFrame, predictions: dict[str, set[str]]) -> pd.DataFrame:
    posting_ids = frame["posting_id"].astype(str)
    return pd.DataFrame(
        {
            "posting_id": posting_ids,
            "matches": [
                " ".join(sorted(predictions[posting_id]))
                for posting_id in posting_ids
            ],
        }
    )


def main() -> None:
    config = parse_args()
    train_path = Path(config.train_csv)
    test_path = Path(config.test_csv)
    output_path = Path(config.output_path)
    reports_dir = Path(config.reports_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_columns(train, {"posting_id", "title"}, train_path)
    validate_columns(test, {"posting_id", "title"}, test_path)

    print(f"Loaded train rows={len(train)} test rows={len(test)}")

    train_metrics: dict[str, float] = {}
    if "label_group" in train.columns and config.diagnostics_limit != 0:
        if 0 < config.diagnostics_limit < len(train):
            diagnostic_frame = train.sample(config.diagnostics_limit, random_state=config.seed)
        else:
            diagnostic_frame = train
        train_predictions = combine_matches(
            exact_phash_matches(diagnostic_frame),
            title_neighbor_matches(diagnostic_frame, config),
        )
        train_metrics = evaluate_matches(diagnostic_frame, train_predictions)
    else:
        diagnostic_frame = train.iloc[0:0]

    if train_metrics:
        print(
            "Train diagnostics: "
            f"mean_f1={train_metrics['mean_f1']:.5f} "
            f"mean_match_count={train_metrics['mean_match_count']:.2f}"
        )

    test_predictions = combine_matches(
        exact_phash_matches(test),
        title_neighbor_matches(test, config),
    )
    submission = format_submission(test, test_predictions)
    submission.to_csv(output_path, index=False)
    print(f"Wrote submission to {output_path}")

    report = {
        "config": asdict(config),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "diagnostic_rows": int(len(diagnostic_frame)),
        "train_metrics": train_metrics,
        "submission_path": str(output_path),
    }
    (reports_dir / "shopee_baseline_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
