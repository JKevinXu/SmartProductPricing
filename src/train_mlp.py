"""Train a text-first MLP baseline and create a Kaggle submission."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import build_numeric_text_features, clean_text, numeric_feature_names
from src.metrics import smape


@dataclass
class TrainConfig:
    train_csv: str
    test_csv: str
    output_dir: str
    model_dir: str
    reports_dir: str
    submission_filename: str
    seed: int
    valid_size: float
    max_features: int
    svd_components: int
    hidden_dims: tuple[int, ...]
    dropout: float
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    min_prediction: float
    device: str


class PriceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--output-dir", default="data/submissions")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--submission-filename", default="mlp_submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--svd-components", type=int, default=256)
    parser.add_argument("--hidden-dims", default="512,256")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-prediction", type=float, default=0.01)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    hidden_dims = tuple(
        int(value.strip())
        for value in args.hidden_dims.split(",")
        if value.strip()
    )
    if not hidden_dims:
        raise ValueError("--hidden-dims must contain at least one integer")

    return TrainConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        submission_filename=args.submission_filename,
        seed=args.seed,
        valid_size=args.valid_size,
        max_features=args.max_features,
        svd_components=args.svd_components,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        min_prediction=args.min_prediction,
        device=args.device,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested_device)


def validate_columns(frame: pd.DataFrame, required_columns: set[str], path: Path) -> None:
    missing = required_columns.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_text}")


def make_stratify_bins(price: pd.Series) -> pd.Series | None:
    if price.nunique() < 10:
        return None
    try:
        bins = pd.qcut(np.log1p(price), q=10, labels=False, duplicates="drop")
    except ValueError:
        return None
    counts = bins.value_counts()
    if counts.empty or counts.min() < 2:
        return None
    return bins


def fit_feature_pipeline(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame | None,
    config: TrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, object]]:
    train_text = clean_text(train_frame["catalog_content"])
    valid_text = clean_text(valid_frame["catalog_content"])
    test_text = clean_text(test_frame["catalog_content"]) if test_frame is not None else None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=1,
        max_features=config.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_tfidf = vectorizer.fit_transform(train_text.replace("", "missing"))
    valid_tfidf = vectorizer.transform(valid_text.replace("", "missing"))
    test_tfidf = vectorizer.transform(test_text.replace("", "missing")) if test_text is not None else None

    effective_components = min(
        config.svd_components,
        max(1, train_tfidf.shape[1] - 1),
        max(1, train_tfidf.shape[0] - 1),
    )
    if effective_components < 2:
        train_dense = train_tfidf.toarray().astype(np.float32)
        valid_dense = valid_tfidf.toarray().astype(np.float32)
        test_dense = test_tfidf.toarray().astype(np.float32) if test_tfidf is not None else None
        svd = None
    else:
        svd = TruncatedSVD(n_components=effective_components, random_state=config.seed)
        train_dense = svd.fit_transform(train_tfidf).astype(np.float32)
        valid_dense = svd.transform(valid_tfidf).astype(np.float32)
        test_dense = svd.transform(test_tfidf).astype(np.float32) if test_tfidf is not None else None

    train_numeric = build_numeric_text_features(train_frame)
    valid_numeric = build_numeric_text_features(valid_frame)
    test_numeric = build_numeric_text_features(test_frame) if test_frame is not None else None

    train_features = np.hstack([train_dense, train_numeric]).astype(np.float32)
    valid_features = np.hstack([valid_dense, valid_numeric]).astype(np.float32)
    test_features = (
        np.hstack([test_dense, test_numeric]).astype(np.float32)
        if test_dense is not None and test_numeric is not None
        else None
    )

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features).astype(np.float32)
    valid_features = scaler.transform(valid_features).astype(np.float32)
    if test_features is not None:
        test_features = scaler.transform(test_features).astype(np.float32)

    artifacts = {
        "vectorizer": vectorizer,
        "svd": svd,
        "scaler": scaler,
        "numeric_feature_names": numeric_feature_names(),
        "svd_components": effective_components,
        "tfidf_shape": train_tfidf.shape,
    }
    return train_features, valid_features, test_features, artifacts


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model: PriceMLP, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item()) * len(targets)
            total_count += len(targets)
    return total_loss / max(total_count, 1)


def predict_log_targets(
    model: PriceMLP,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    dataset = TensorDataset(torch.from_numpy(features.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_features,) in loader:
            outputs = model(batch_features.to(device))
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions)


def inverse_price(log_predictions: np.ndarray, min_prediction: float) -> np.ndarray:
    return np.clip(np.expm1(log_predictions), min_prediction, None)


def train_model(
    train_features: np.ndarray,
    valid_features: np.ndarray,
    y_train_log: np.ndarray,
    y_valid_log: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> tuple[PriceMLP, list[dict[str, float]]]:
    model = PriceMLP(train_features.shape[1], config.hidden_dims, config.dropout).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    train_loader = make_loader(train_features, y_train_log, config.batch_size, shuffle=True)
    valid_loader = make_loader(valid_features, y_valid_log, config.batch_size, shuffle=False)

    best_state = None
    best_valid_loss = math.inf
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(targets)
            total_count += len(targets)

        train_loss = total_loss / max(total_count, 1)
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"valid_loss={valid_loss:.5f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save_report(
    report_path: Path,
    config: TrainConfig,
    metrics: dict[str, float],
    history: list[dict[str, float]],
    feature_artifacts: dict[str, object],
) -> None:
    report = {
        "config": asdict(config),
        "metrics": metrics,
        "history": history,
        "features": {
            "tfidf_shape": list(feature_artifacts["tfidf_shape"]),
            "svd_components": feature_artifacts["svd_components"],
            "numeric_feature_names": feature_artifacts["numeric_feature_names"],
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    device = resolve_device(config.device)

    train_path = Path(config.train_csv)
    test_path = Path(config.test_csv)
    output_dir = Path(config.output_dir)
    model_dir = Path(config.model_dir)
    reports_dir = Path(config.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found at {train_path}. "
            "Put train.csv under data/raw or pass --train-csv."
        )

    train_full = pd.read_csv(train_path)
    validate_columns(train_full, {"sample_id", "catalog_content", "price"}, train_path)
    train_full = train_full[train_full["price"].notna()].copy()
    train_full["price"] = pd.to_numeric(train_full["price"], errors="coerce")
    train_full = train_full[train_full["price"].notna() & (train_full["price"] > 0)].copy()
    if len(train_full) < 2:
        raise ValueError("Need at least two rows with positive prices to create a validation split.")

    stratify_bins = make_stratify_bins(train_full["price"])
    train_frame, valid_frame = train_test_split(
        train_full,
        test_size=config.valid_size,
        random_state=config.seed,
        stratify=stratify_bins,
    )

    test_frame = None
    if test_path.exists():
        test_frame = pd.read_csv(test_path)
        validate_columns(test_frame, {"sample_id", "catalog_content"}, test_path)
    else:
        print(f"Test CSV not found at {test_path}; training validation model only.")

    print(
        f"Training rows={len(train_frame)} validation rows={len(valid_frame)} "
        f"test rows={0 if test_frame is None else len(test_frame)} device={device}"
    )

    train_features, valid_features, test_features, feature_artifacts = fit_feature_pipeline(
        train_frame,
        valid_frame,
        test_frame,
        config,
    )

    y_train_log = np.log1p(train_frame["price"].to_numpy(dtype=np.float32))
    y_valid_log = np.log1p(valid_frame["price"].to_numpy(dtype=np.float32))

    model, history = train_model(
        train_features,
        valid_features,
        y_train_log,
        y_valid_log,
        config,
        device,
    )

    valid_log_predictions = predict_log_targets(model, valid_features, config.batch_size, device)
    valid_predictions = inverse_price(valid_log_predictions, config.min_prediction)
    valid_prices = valid_frame["price"].to_numpy(dtype=np.float64)
    metrics = {
        "valid_smape": smape(valid_prices, valid_predictions),
        "valid_mae": float(np.mean(np.abs(valid_predictions - valid_prices))),
        "valid_rmse": float(np.sqrt(np.mean((valid_predictions - valid_prices) ** 2))),
    }
    print(
        f"Validation SMAPE={metrics['valid_smape']:.4f} "
        f"MAE={metrics['valid_mae']:.4f} RMSE={metrics['valid_rmse']:.4f}"
    )

    validation_predictions_path = reports_dir / "mlp_validation_predictions.csv"
    pd.DataFrame(
        {
            "sample_id": valid_frame["sample_id"].to_numpy(),
            "price": valid_prices,
            "predicted_price": valid_predictions,
        }
    ).to_csv(validation_predictions_path, index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": train_features.shape[1],
            "config": asdict(config),
        },
        model_dir / "mlp_model.pt",
    )
    joblib.dump(feature_artifacts, model_dir / "mlp_preprocessor.joblib")
    save_report(reports_dir / "mlp_report.json", config, metrics, history, feature_artifacts)

    if test_frame is not None and test_features is not None:
        test_log_predictions = predict_log_targets(model, test_features, config.batch_size, device)
        test_predictions = inverse_price(test_log_predictions, config.min_prediction)
        submission = pd.DataFrame(
            {
                "sample_id": test_frame["sample_id"].to_numpy(),
                "price": test_predictions,
            }
        )
        submission_path = output_dir / config.submission_filename
        submission.to_csv(submission_path, index=False)
        print(f"Wrote submission to {submission_path}")


if __name__ == "__main__":
    main()
