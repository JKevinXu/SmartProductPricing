from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
USE_PRETRAINED_IMAGE_EMBEDDINGS = False
PRETRAINED_MODEL = "clip_vit_b32"
PRETRAINED_IMAGE_THRESHOLD = 0.75
MAX_PRETRAINED_IMAGE_NEIGHBORS = 80
PRETRAINED_IMAGE_SIZE = 224
PRETRAINED_BATCH_SIZE = 64
PRETRAINED_WEIGHTS_FILE = "ViT-B-32.pt"
USE_PAIRWISE_RERANKER = False
RERANK_CANDIDATE_THRESHOLD = 0.35
RERANK_PROBA_THRESHOLD = 0.30
RERANK_MAX_TRAIN_PAIRS = 240000
RERANK_NEGATIVE_RATIO = 4

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


def title_matrices(train: pd.DataFrame, test: pd.DataFrame):
    combined_titles = pd.concat([normalize_titles(train["title"]), normalize_titles(test["title"])], ignore_index=True)
    split = len(train)

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=MAX_CHAR_FEATURES,
        lowercase=False,
        dtype=np.float32,
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
        max_features=MAX_WORD_FEATURES,
        lowercase=False,
        dtype=np.float32,
    )
    char_matrix = char_vectorizer.fit_transform(combined_titles)
    word_matrix = word_vectorizer.fit_transform(combined_titles)
    return (
        char_matrix[:split],
        char_matrix[split:],
        word_matrix[:split],
        word_matrix[split:],
    )


def neighbor_pair_scores(matrix, threshold: float, max_neighbors: int) -> dict[tuple[int, int], float]:
    neighbor_count = min(max_neighbors, matrix.shape[0])
    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix, return_distance=True)

    scores: dict[tuple[int, int], float] = {}
    for row_index in range(matrix.shape[0]):
        for distance, neighbor_index in zip(distances[row_index], indices[row_index], strict=False):
            if row_index == neighbor_index:
                continue
            similarity = float(1.0 - distance)
            if similarity < threshold:
                continue
            pair = tuple(sorted((row_index, int(neighbor_index))))
            if similarity > scores.get(pair, 0.0):
                scores[pair] = similarity
    return scores


def phash_pair_scores(frame: pd.DataFrame) -> dict[tuple[int, int], float]:
    if "image_phash" not in frame.columns:
        return {}
    scores: dict[tuple[int, int], float] = {}
    for _, group in frame.groupby("image_phash", dropna=False):
        indices = group.index.to_list()
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                scores[tuple(sorted((int(left_index), int(right_index))))] = 1.0
    return scores


def title_tokens(frame: pd.DataFrame) -> list[set[str]]:
    return [set(title.split()) for title in normalize_titles(frame["title"])]


def number_tokens(tokens: set[str]) -> set[str]:
    return {token for token in tokens if any(character.isdigit() for character in token)}


def pair_feature_rows(
    frame: pd.DataFrame,
    pairs: list[tuple[int, int]],
    char_scores: dict[tuple[int, int], float],
    word_scores: dict[tuple[int, int], float],
    phash_scores: dict[tuple[int, int], float],
) -> np.ndarray:
    tokens = title_tokens(frame)
    numbers = [number_tokens(token_set) for token_set in tokens]
    phashes = frame["image_phash"].astype(str).fillna("").to_numpy() if "image_phash" in frame.columns else None
    rows = []
    for left, right in pairs:
        pair = tuple(sorted((left, right)))
        left_tokens = tokens[left]
        right_tokens = tokens[right]
        union_count = len(left_tokens.union(right_tokens))
        title_jaccard = len(left_tokens.intersection(right_tokens)) / union_count if union_count else 0.0
        left_len = max(len(left_tokens), 1)
        right_len = max(len(right_tokens), 1)
        length_ratio = min(left_len, right_len) / max(left_len, right_len)
        left_numbers = numbers[left]
        right_numbers = numbers[right]
        number_union = len(left_numbers.union(right_numbers))
        number_jaccard = len(left_numbers.intersection(right_numbers)) / number_union if number_union else 0.0
        same_phash = 0.0
        if phashes is not None and phashes[left] and phashes[left] == phashes[right]:
            same_phash = 1.0
        char_similarity = char_scores.get(pair, 0.0)
        word_similarity = word_scores.get(pair, 0.0)
        rows.append(
            [
                same_phash,
                char_similarity,
                word_similarity,
                max(char_similarity, word_similarity),
                title_jaccard,
                length_ratio,
                number_jaccard,
                float(pair in phash_scores),
                float(pair in char_scores),
                float(pair in word_scores),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def same_label_array(frame: pd.DataFrame, pairs: list[tuple[int, int]]) -> np.ndarray:
    labels = frame["label_group"].to_numpy()
    return np.asarray([labels[left] == labels[right] for left, right in pairs], dtype=np.int32)


def label_positive_pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
    if "label_group" not in frame.columns:
        return set()
    pairs: set[tuple[int, int]] = set()
    for _, group in frame.groupby("label_group"):
        indices = group.index.to_list()
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                pairs.add(tuple(sorted((int(left_index), int(right_index)))))
    return pairs


def sampled_training_pairs(frame: pd.DataFrame, candidate_pairs: set[tuple[int, int]], seed: int = 42) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    pairs = list(candidate_pairs)
    labels = same_label_array(frame, pairs)
    positives = [pair for pair, label in zip(pairs, labels, strict=False) if label == 1]
    negatives = [pair for pair, label in zip(pairs, labels, strict=False) if label == 0]

    if positives:
        positive_limit = min(
            len(positives),
            max(RERANK_MAX_TRAIN_PAIRS // (RERANK_NEGATIVE_RATIO + 1), 1),
        )
        negative_limit = min(
            len(negatives),
            max(positive_limit * RERANK_NEGATIVE_RATIO, 1),
        )
    else:
        positive_limit = 0
        negative_limit = min(len(negatives), RERANK_MAX_TRAIN_PAIRS)

    if positive_limit < len(positives):
        positive_indices = rng.choice(len(positives), size=positive_limit, replace=False)
        positives = [positives[index] for index in positive_indices]
    if negative_limit < len(negatives):
        negative_indices = rng.choice(len(negatives), size=negative_limit, replace=False)
        negatives = [negatives[index] for index in negative_indices]

    sampled = positives + negatives
    rng.shuffle(sampled)
    return sampled


def train_pairwise_reranker(
    train: pd.DataFrame,
    train_char_scores: dict[tuple[int, int], float],
    train_word_scores: dict[tuple[int, int], float],
    train_phash_scores: dict[tuple[int, int], float],
) -> LogisticRegression | None:
    if "label_group" not in train.columns:
        return None
    candidate_pairs = set(train_char_scores).union(train_word_scores).union(train_phash_scores)
    candidate_pairs.update(label_positive_pairs(train))
    if not candidate_pairs:
        return None
    train_pairs = sampled_training_pairs(train, candidate_pairs)
    if not train_pairs:
        return None
    y = same_label_array(train, train_pairs)
    if len(np.unique(y)) < 2:
        return None
    x = pair_feature_rows(train, train_pairs, train_char_scores, train_word_scores, train_phash_scores)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42,
    )
    model.fit(x, y)
    print(
        "Trained reranker "
        f"pairs={len(train_pairs)} positives={int(y.sum())} negatives={int((1 - y).sum())}"
    )
    return model


def reranked_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, set[str]]:
    train_char_matrix, test_char_matrix, train_word_matrix, test_word_matrix = title_matrices(train, test)
    train_char_scores = neighbor_pair_scores(
        train_char_matrix,
        RERANK_CANDIDATE_THRESHOLD,
        MAX_CHAR_TITLE_NEIGHBORS,
    )
    train_word_scores = neighbor_pair_scores(
        train_word_matrix,
        RERANK_CANDIDATE_THRESHOLD,
        MAX_WORD_TITLE_NEIGHBORS,
    )
    train_phash_scores = phash_pair_scores(train)
    model = train_pairwise_reranker(train, train_char_scores, train_word_scores, train_phash_scores)
    if model is None:
        return combine_matches(exact_phash_matches(test), char_title_matches(test), word_title_matches(test))

    test_char_scores = neighbor_pair_scores(
        test_char_matrix,
        RERANK_CANDIDATE_THRESHOLD,
        MAX_CHAR_TITLE_NEIGHBORS,
    )
    test_word_scores = neighbor_pair_scores(
        test_word_matrix,
        RERANK_CANDIDATE_THRESHOLD,
        MAX_WORD_TITLE_NEIGHBORS,
    )
    test_phash_scores = phash_pair_scores(test)
    candidate_pairs = sorted(set(test_char_scores).union(test_word_scores).union(test_phash_scores))

    posting_ids = test["posting_id"].astype(str).to_numpy()
    predictions = {posting_id: {posting_id} for posting_id in posting_ids}
    for left, right in test_phash_scores:
        predictions[posting_ids[left]].add(posting_ids[right])
        predictions[posting_ids[right]].add(posting_ids[left])

    if not candidate_pairs:
        return predictions

    features = pair_feature_rows(test, candidate_pairs, test_char_scores, test_word_scores, test_phash_scores)
    probabilities = model.predict_proba(features)[:, 1]
    kept = 0
    for probability, (left, right) in zip(probabilities, candidate_pairs, strict=False):
        if probability >= RERANK_PROBA_THRESHOLD:
            predictions[posting_ids[left]].add(posting_ids[right])
            predictions[posting_ids[right]].add(posting_ids[left])
            kept += 1
    print(f"Reranked test pairs candidates={len(candidate_pairs)} kept={kept}")
    return predictions


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


def find_train_csv() -> Path | None:
    candidates = []
    for root in [KAGGLE_INPUT_DIR, Path("/kaggle/input"), LOCAL_INPUT_DIR]:
        if root.exists():
            candidates.extend(root.rglob("train.csv"))

    for candidate in candidates:
        try:
            columns = pd.read_csv(candidate, nrows=0).columns
        except Exception:
            continue
        if {"posting_id", "title", "label_group"}.issubset(set(columns)):
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
    train_path = find_train_csv()
    if USE_PAIRWISE_RERANKER and train_path is not None:
        print(f"Using train CSV for reranker: {train_path}")
        train = pd.read_csv(train_path)
        predictions = reranked_predictions(train, test)
    else:
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
