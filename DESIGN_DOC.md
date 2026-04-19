# Smart Product Pricing Design Doc

## 1. Overview

SmartProductPricing is a Kaggle-style practice project based on the Amazon ML Challenge 2025 Smart Product Pricing task. The goal is to predict product prices from catalog text and product images.

This project is designed as a realistic e-commerce ML workflow:

- Build a strong text-only baseline.
- Add structured feature extraction from product catalog text.
- Add image embeddings for multimodal modeling.
- Explore LLM-assisted feature extraction where it improves signal without leaking prices.
- Produce reproducible validation metrics and a submission-ready `sample_id,price` CSV.

## 2. Problem Statement

Given a product listing with catalog text and an image link, predict the product price.

Input fields:

- `sample_id`: unique product identifier.
- `catalog_content`: product title, description, bullet points, quantity, and related listing text.
- `image_link`: source URL for product image.

Target:

- `price`: positive floating-point product price, available only for training rows.

Output:

- CSV with columns `sample_id` and `price`.
- One prediction for every test row.
- Predicted prices must be positive floats.

## 3. Success Metric

The target metric is SMAPE, Symmetric Mean Absolute Percentage Error.

Lower is better.

```text
SMAPE = mean(abs(predicted - actual) / ((abs(actual) + abs(predicted)) / 2)) * 100
```

Important implications:

- Low-priced products are easy to over-penalize.
- Predicting negative prices is invalid and must be prevented.
- Modeling `log1p(price)` is likely more stable than modeling raw price.
- Validation should report both SMAPE and error slices by price bucket.

## 4. Data Strategy

Expected data layout:

```text
data/
  raw/
    train.csv
    test.csv
    sample_test.csv
    sample_test_out.csv
    train_images/
    test_images/
  processed/
  features/
  submissions/
```

Large files should not be committed to Git. The repo should keep code, docs, configs, and lightweight metadata only.

Recommended `.gitignore` entries:

```text
data/raw/
data/processed/
data/features/
data/submissions/
models/
*.parquet
*.pkl
*.joblib
*.npy
*.npz
```

## 5. Validation Design

Use a local validation split because the original competition leaderboard is not active.

Baseline split:

- 80 percent train.
- 20 percent validation.
- Random split with a fixed seed.

Improved split:

- Stratify by `log1p(price)` quantile buckets.
- Keep rare high-price products represented in validation.

Validation outputs:

- Overall SMAPE.
- SMAPE by price quantile.
- SMAPE by inferred category.
- SMAPE by pack-size availability.
- Example worst predictions for manual inspection.

## 6. Baseline Model

The first target is a fast, reproducible text-only neural baseline.

Feature set:

- TF-IDF word n-grams from `catalog_content`.
- TF-IDF character n-grams from `catalog_content`.
- Basic numeric text features:
  - character count
  - word count
  - digit count
  - uppercase ratio
  - punctuation count
  - detected quantity count

Candidate models:

- MLP on dense TF-IDF/SVD text features plus numeric text features.
- Ridge regression on `log1p(price)`.
- LightGBM on sparse plus dense features.
- CatBoost with text features if local setup supports it.

Baseline acceptance target:

- Pipeline runs end-to-end locally.
- Produces validation SMAPE.
- Produces a valid submission CSV.

## 7. Structured Feature Extraction

Product pricing often depends on structured attributes hidden inside text. The project should extract them before investing in heavier models.

Regex/parser features:

- Brand candidate from title prefix.
- Pack quantity, such as `pack of 6`, `6 pack`, `12-count`.
- Unit quantity, such as ounces, pounds, grams, ml, liters, count, pieces.
- Normalized unit amount where possible.
- Product category keywords.
- Size descriptors, such as small, medium, large, jumbo, family size.
- Premium descriptors, such as organic, gluten-free, stainless steel, leather.

Derived features:

- `unit_price_proxy = predicted_price / normalized_quantity` for analysis only.
- Binary flags for detected quantity, detected brand, detected unit, and detected category.
- Frequency-encoded brand/category features.

## 8. LLM-Assisted Features

LLMs can help convert messy catalog text into structured product attributes. They should not be used to look up real-world prices or enrich data from external product pages.

Allowed LLM uses:

- Extract structured attributes from `catalog_content`.
- Normalize units.
- Infer broad product category.
- Generate concise product summaries.
- Identify likely brand and product type.
- Explain validation failures for model debugging.

Disallowed uses:

- Web search for current or historical product prices.
- External product lookup by image URL, title, ASIN, or exact product name.
- Any price labels or price-derived external data not present in the training set.

Suggested JSON extraction schema:

```json
{
  "brand": "string_or_null",
  "product_type": "string",
  "category": "string",
  "pack_count": "number_or_null",
  "unit_value": "number_or_null",
  "unit_name": "string_or_null",
  "material_or_flavor": "string_or_null",
  "premium_flags": ["string"],
  "target_customer": "string_or_null"
}
```

Implementation guidance:

- Start with a small sample of rows to validate extraction quality.
- Cache all LLM outputs under `data/features/llm_attributes.parquet`.
- Include prompt version, model name, and extraction timestamp.
- Never call the LLM during model training if cached features are available.

## 9. Image Feature Strategy

Product images provide signal for category, packaging, brand, and product size.

Initial image plan:

- Download or use pre-downloaded images.
- Verify image coverage by matching filenames to `image_link`.
- Generate image embeddings with a pretrained CLIP-style model.
- Save embeddings to `data/features/image_embeddings.parquet` or `.npy`.

Image model usage:

- Do not train a vision model from scratch initially.
- Use frozen embeddings as features.
- Combine image embeddings with text and structured features in gradient boosting or a small neural model.

Failure handling:

- Missing images should map to zero vectors plus a `missing_image` flag.
- Corrupt images should be logged and skipped without crashing the full feature job.

## 10. Modeling Roadmap

Phase 1: Text Baseline

- Load CSV data.
- Clean `catalog_content`.
- Train TF-IDF plus SVD plus MLP on `log1p(price)`.
- Evaluate SMAPE.
- Save predictions and validation report.

Phase 2: Better Tabular/Text Model

- Add regex-derived structured features.
- Train LightGBM or CatBoost.
- Compare against Ridge baseline.
- Add price-bucket diagnostics.

Phase 3: LLM Attributes

- Extract structured attributes for train/test.
- Add cached LLM features to model.
- Measure whether they improve validation SMAPE.
- Keep only features that improve validation or diagnostics.

Phase 4: Multimodal Model

- Generate image embeddings.
- Combine text, structured, LLM, and image features.
- Train LightGBM/CatBoost ensemble.
- Compare against text-only model.

Phase 5: Ensemble

- Blend Ridge, LightGBM, CatBoost, and optional neural predictions.
- Tune blend weights on validation.
- Generate final submission.

## 11. Reproducibility

All scripts should accept explicit input/output paths and random seeds.

Recommended commands:

```bash
python -m src.train_baseline --seed 42
python -m src.extract_text_features
python -m src.extract_llm_features --limit 100
python -m src.extract_image_features
python -m src.make_submission --model-path models/best.joblib
```

Artifacts:

- `models/`: trained models.
- `data/features/`: generated features.
- `reports/`: validation metrics and plots.
- `data/submissions/`: submission CSVs.

## 12. Risks And Mitigations

Risk: Dataset is large and image processing may be slow.

Mitigation: Start text-only, then add cached image embeddings in batches.

Risk: LLM feature extraction can be expensive.

Mitigation: Validate on a small sample, cache outputs, and use LLMs only where regex features are weak.

Risk: SMAPE can reward conservative predictions and punish low-price misses.

Mitigation: Train on `log1p(price)`, clip predictions to positive values, and inspect low-price buckets.

Risk: Data leakage from external product lookup.

Mitigation: Only use provided dataset fields and pretrained general-purpose models. Do not search product prices online.

## 13. Initial Milestones

1. Create project scaffold and `.gitignore`.
2. Add data loading and validation split.
3. Implement SMAPE metric.
4. Train text-only baseline.
5. Add regex quantity/unit/brand features.
6. Add validation report with price-bucket analysis.
7. Add optional LLM extraction cache.
8. Add image embedding pipeline.
9. Train ensemble and generate final submission.

## 14. Open Questions

- Which Kaggle dataset mirror will be used as the canonical local data source?
- Should the first implementation optimize for local laptop runtime or best possible score?
- Which LLM provider/model should be used for structured extraction?
- Should image features be included in the first milestone or deferred until the text baseline is stable?
