# SmartProductPricing

Kaggle-style Smart Product Pricing project. The first runnable model is a
text-first MLP baseline that predicts `log1p(price)` and writes a
submission-ready CSV with `sample_id,price`.

## Data Layout

Place the competition CSVs here:

```text
data/
  raw/
    train.csv
    test.csv
```

Expected columns:

- `train.csv`: `sample_id`, `catalog_content`, `image_link`, `price`
- `test.csv`: `sample_id`, `catalog_content`, `image_link`

`image_link` is optional for the first MLP, but it is used as a simple
availability flag when present.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train The MLP

```bash
python3 -m src.train_mlp \
  --train-csv data/raw/train.csv \
  --test-csv data/raw/test.csv \
  --epochs 30 \
  --batch-size 512
```

Outputs:

- `reports/mlp_report.json`: validation metrics and training history
- `reports/mlp_validation_predictions.csv`: held-out predictions for inspection
- `models/mlp_model.pt`: trained PyTorch model
- `models/mlp_preprocessor.joblib`: TF-IDF, SVD, scaler, and feature metadata
- `data/submissions/mlp_submission.csv`: Kaggle-ready submission

## Fast Smoke Run

For a quick local check on a small slice:

```bash
python3 -m src.train_mlp --epochs 2 --max-features 5000 --svd-components 64
```

## Notes

The model uses:

- TF-IDF word n-grams from `catalog_content`
- Truncated SVD to convert sparse text features into dense MLP inputs
- Basic numeric text features such as length, digit counts, pack/unit hints
- A PyTorch MLP trained with Smooth L1 loss on `log1p(price)`
- Positive clipping before SMAPE evaluation and submission export

## Shopee Product Matching

Shopee is a product matching task, not a price prediction task. The baseline
for it lives in `src/shopee_baseline.py` and outputs Kaggle's required
`posting_id,matches` format.

Download the competition data:

```bash
mkdir -p data/shopee/raw
kaggle competitions download -c shopee-product-matching -p data/shopee/raw --unzip
```

Run the baseline:

```bash
python3 -m src.shopee_baseline \
  --train-csv data/shopee/raw/train.csv \
  --test-csv data/shopee/raw/test.csv \
  --output-path data/submissions/shopee_submission.csv
```

The baseline combines exact `image_phash` groups with character and word-level
title TF-IDF nearest neighbors, and writes capped local F1 diagnostics to
`reports/shopee_baseline_report.json` when `label_group` is available. Use
`--diagnostics-limit 0` to skip diagnostics or a larger value for a slower,
broader local check.

The local script also has an opt-in handcrafted image-vector experiment through
`--image-embeddings`. That experiment scored worse on Kaggle than the text and
pHash baseline, so the submitted kernel leaves it disabled.

The next image experiment uses offline CLIP ViT-B/32 weights attached to the
Kaggle kernel as `kevinxuj/shopee-clip-vit-b32-weights`. Locally, enable it
with:

```bash
python3 -m src.shopee_baseline \
  --train-csv data/shopee/raw/train.csv \
  --test-csv data/shopee/raw/test.csv \
  --pretrained-image-embeddings \
  --pretrained-model clip_vit_b32 \
  --pretrained-weights-path data/shopee/raw/clip-vit-b32-weights/ViT-B-32.pt
```
