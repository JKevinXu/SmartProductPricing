from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
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
USE_PRETRAINED_IMAGE_EMBEDDINGS = True
PRETRAINED_MODEL = "clip_vit_b32"
PRETRAINED_IMAGE_THRESHOLD = 0.75
MAX_PRETRAINED_IMAGE_NEIGHBORS = 80
PRETRAINED_IMAGE_SIZE = 224
PRETRAINED_BATCH_SIZE = 64
PRETRAINED_WEIGHTS_FILE = "ViT-B-32.pt"

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


def find_image_dir(test_path: Path | None) -> Path | None:
    candidates = []
    if test_path is not None:
        candidates.append(test_path.parent / "test_images")
    candidates.extend(
        [
            INPUT_DIR / "test_images",
            Path("/kaggle/input/competitions/shopee-product-matching/test_images"),
            LOCAL_INPUT_DIR / "test_images",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def find_pretrained_weights() -> Path | None:
    candidates = [
        Path("/kaggle/input/shopee-clip-vit-b32-weights") / PRETRAINED_WEIGHTS_FILE,
        LOCAL_INPUT_DIR / "clip-vit-b32-weights" / PRETRAINED_WEIGHTS_FILE,
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.rglob(PRETRAINED_WEIGHTS_FILE))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_resnet18_feature_model(weights_path: Path):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    feature_model = nn.Sequential(*list(model.children())[:-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_model.to(device)
    feature_model.eval()
    return feature_model, device


def resnet18_image_tensor(image_path: Path):
    import torch

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((PRETRAINED_IMAGE_SIZE, PRETRAINED_IMAGE_SIZE), Image.Resampling.BILINEAR)
    except Exception:
        return None

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )
    return torch.from_numpy(pixels).permute(2, 0, 1)


def resnet18_image_embeddings(frame: pd.DataFrame, image_dir: Path, weights_path: Path) -> np.ndarray:
    import torch

    model, device = load_resnet18_feature_model(weights_path)
    embeddings = np.zeros((len(frame), 512), dtype=np.float32)
    batch_tensors = []
    batch_indices = []

    def flush_batch() -> None:
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            features = model(batch).flatten(1)
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        embeddings[batch_indices] = features.cpu().numpy().astype(np.float32)
        batch_tensors.clear()
        batch_indices.clear()

    for row_index, image_name in enumerate(frame["image"].fillna("")):
        tensor = resnet18_image_tensor(image_dir / str(image_name))
        if tensor is None:
            continue
        batch_tensors.append(tensor)
        batch_indices.append(row_index)
        if len(batch_tensors) >= PRETRAINED_BATCH_SIZE:
            flush_batch()
    flush_batch()
    return embeddings


def load_clip_feature_model(weights_path: Path):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CLIP JIT weights require a CUDA runtime in this kernel")
    model = torch.jit.load(str(weights_path), map_location=device).eval()
    return model, device


def clip_image_tensor(image_path: Path):
    import torch

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.fit(
                image,
                (PRETRAINED_IMAGE_SIZE, PRETRAINED_IMAGE_SIZE),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
    except Exception:
        return None

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)) / np.array(
        [0.26862954, 0.26130258, 0.27577711],
        dtype=np.float32,
    )
    return torch.from_numpy(pixels).permute(2, 0, 1)


def clip_image_embeddings(frame: pd.DataFrame, image_dir: Path, weights_path: Path) -> np.ndarray:
    import torch

    model, device = load_clip_feature_model(weights_path)
    embeddings = np.zeros((len(frame), 512), dtype=np.float32)
    batch_tensors = []
    batch_indices = []

    def flush_batch() -> None:
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            features = model.encode_image(batch).float()
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        embeddings[batch_indices] = features.cpu().numpy().astype(np.float32)
        batch_tensors.clear()
        batch_indices.clear()

    for row_index, image_name in enumerate(frame["image"].fillna("")):
        tensor = clip_image_tensor(image_dir / str(image_name))
        if tensor is None:
            continue
        batch_tensors.append(tensor)
        batch_indices.append(row_index)
        if len(batch_tensors) >= PRETRAINED_BATCH_SIZE:
            flush_batch()
    flush_batch()
    return embeddings


def pretrained_image_matches(
    frame: pd.DataFrame,
    image_dir: Path | None,
    weights_path: Path | None,
) -> dict[str, set[str]]:
    if not USE_PRETRAINED_IMAGE_EMBEDDINGS:
        return {str(posting_id): {str(posting_id)} for posting_id in frame["posting_id"]}
    if image_dir is None or weights_path is None or "image" not in frame.columns:
        print("Skipping pretrained image embeddings; image directory or weights are missing.")
        return {str(posting_id): {str(posting_id)} for posting_id in frame["posting_id"]}

    try:
        if PRETRAINED_MODEL == "clip_vit_b32":
            embeddings = clip_image_embeddings(frame, image_dir, weights_path)
        elif PRETRAINED_MODEL == "resnet18":
            embeddings = resnet18_image_embeddings(frame, image_dir, weights_path)
        else:
            raise ValueError(f"Unknown pretrained model: {PRETRAINED_MODEL}")
    except Exception as exc:
        print(f"Skipping pretrained image embeddings after error: {exc}")
        return {str(posting_id): {str(posting_id)} for posting_id in frame["posting_id"]}

    valid = np.linalg.norm(embeddings, axis=1) > 0
    neighbor_count = min(MAX_PRETRAINED_IMAGE_NEIGHBORS, len(frame))
    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings, return_distance=True)

    posting_ids = frame["posting_id"].astype(str).to_numpy()
    matches: dict[str, set[str]] = {}
    for row_index, posting_id in enumerate(posting_ids):
        row_matches = {posting_id}
        if valid[row_index]:
            similarities = 1.0 - distances[row_index]
            for similarity, neighbor_index in zip(similarities, indices[row_index], strict=False):
                if valid[neighbor_index] and similarity >= PRETRAINED_IMAGE_THRESHOLD:
                    row_matches.add(str(posting_ids[neighbor_index]))
        matches[str(posting_id)] = row_matches
    return matches


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
    image_dir = find_image_dir(test_path)
    weights_path = find_pretrained_weights()
    print(f"Using image dir: {image_dir}")
    print(f"Using pretrained weights: {weights_path}")
    predictions = combine_matches(
        exact_phash_matches(test),
        char_title_matches(test),
        word_title_matches(test),
        pretrained_image_matches(test, image_dir, weights_path),
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
